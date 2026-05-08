import copy
import os
from collections import OrderedDict

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .common_utils import CommonUtils
from .mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from .track_utils import sample_points_from_masks
from .video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class GroundingDinoPredictor:
    """Wrapper for GroundingDINO zero-shot object detection."""

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def predict(self, image, text_prompts: str, box_threshold=0.25, text_threshold=0.25):
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        # Parse the original prompt back into individual tags so we can snap
        # each raw GroundingDINO label to the nearest prompt tag.  This prevents
        # span-bleed where adjacent similar tags (e.g. "red cup" / "red mug")
        # get merged into a single blended label like "red cup mug".
        prompt_tags = [t.strip().rstrip(".") for t in text_prompts.split(".") if t.strip()]

        def _snap_to_tag(raw_label: str) -> str:
            """Return the prompt tag whose tokens overlap most with raw_label."""
            raw_tokens = set(raw_label.lower().split())
            best_tag, best_score = raw_label, -1
            for tag in prompt_tags:
                tag_tokens = set(tag.lower().split())
                score = len(raw_tokens & tag_tokens)
                if score > best_score:
                    best_score, best_tag = score, tag
            return best_tag

        labels = [_snap_to_tag(l) for l in results[0]["labels"]]
        return results[0]["boxes"], labels


class SAM2ImageSegmentor:
    """Wrapper for SAM2 segmentation from bounding boxes."""

    def __init__(self, sam_model_cfg: str, sam_model_ckpt: str, device="cuda"):
        self.device = device
        sam_model = build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks, scores, logits


class IncrementalObjectTracker:
    def __init__(
        self,
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text="car.",
        detection_interval=20,
        score_threshold=0.5,
        overlap_iou_threshold=0.5,
    ):
        self.device = device
        self.detection_interval = detection_interval
        self.prompt_text = prompt_text
        self.score_threshold = score_threshold
        self.overlap_iou_threshold = overlap_iou_threshold

        self.grounding_predictor = GroundingDinoPredictor(model_id=grounding_model_id, device=device)
        self.sam2_segmentor = SAM2ImageSegmentor(
            sam_model_cfg=sam2_model_cfg,
            sam_model_ckpt=sam2_ckpt_path,
            device=device,
        )
        self.video_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_ckpt_path)

        self.inference_state = self._make_inference_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.total_frames = 0
        self.objects_count = 0
        self.frame_cache_limit = detection_interval - 1

        self.last_mask_dict = MaskDictionaryModel()
        self.track_dict = MaskDictionaryModel()

    def _make_inference_state(self) -> dict:
        """Build an empty SAM2 inference state without requiring a video file."""
        compute_device = self.video_predictor.device
        state = {
            "images": None,
            "num_frames": 0,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": None,
            "video_width": None,
            "device": compute_device,
            "storage_device": compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        return state

    # SAM2 1.1 removed add_new_frame / infer_single_frame. These two helpers
    # replicate the old behaviour using the still-available internal methods.

    def _add_new_frame(self, image_np: np.ndarray) -> int:
        """Preprocess image_np, append to inference_state["images"], return frame_idx."""
        img_size = self.video_predictor.image_size
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        img_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        pil = Image.fromarray(image_np).resize((img_size, img_size))
        t = torch.from_numpy(np.array(pil)).float() / 255.0  # (H, W, 3)
        t = t.permute(2, 0, 1)                               # (3, H, W)
        t = (t - img_mean) / img_std

        self.inference_state["images"] = torch.cat(
            [self.inference_state["images"], t.unsqueeze(0).to(self.device)], dim=0
        )
        frame_idx = self.inference_state["num_frames"]
        self.inference_state["num_frames"] += 1
        return frame_idx

    @torch.inference_mode()
    def _infer_single_frame(self, frame_idx: int):
        """Run tracking for one frame; returns (frame_idx, obj_ids, masks tensor)."""
        batch_size = self.video_predictor._get_obj_num(self.inference_state)
        if batch_size == 0:
            return frame_idx, [], torch.zeros(0, device=self.device)

        obj_ids = self.inference_state["obj_ids"]
        pred_masks_per_obj = []

        for obj_idx in range(batch_size):
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_output_dict.get("cond_frame_outputs", {}):
                pred_masks = obj_output_dict["cond_frame_outputs"][frame_idx][
                    "pred_masks"
                ].to(self.inference_state["device"])
            else:
                current_out, pred_masks = self.video_predictor._run_single_frame_inference(
                    inference_state=self.inference_state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                obj_output_dict.setdefault("non_cond_frame_outputs", {})[frame_idx] = current_out

            self.inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                "reverse": False
            }
            pred_masks_per_obj.append(pred_masks)

        any_res_masks = torch.cat(pred_masks_per_obj, dim=0)
        _, video_res_masks = self.video_predictor._get_orig_video_res_output(
            self.inference_state, any_res_masks
        )
        return frame_idx, obj_ids, video_res_masks

    def add_image(self, image_np: np.ndarray):
        img_pil = Image.fromarray(image_np)

        if self.total_frames % self.detection_interval == 0:
            if (
                self.inference_state["video_height"] is None
                or self.inference_state["video_width"] is None
            ):
                self.inference_state["video_height"], self.inference_state["video_width"] = image_np.shape[:2]

            if self.inference_state["images"].shape[0] > self.frame_cache_limit:
                print(f"[Reset] Resetting inference state after {self.frame_cache_limit} frames.")
                self.inference_state = self._make_inference_state()
                self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
                self.inference_state["video_height"], self.inference_state["video_width"] = image_np.shape[:2]

            boxes, labels = self.grounding_predictor.predict(img_pil, self.prompt_text)
            if boxes.shape[0] == 0:
                return None

            self.sam2_segmentor.set_image(image_np)
            masks, scores, logits = self.sam2_segmentor.predict_masks_from_boxes(boxes)

            # scores shape: (N,) — one SAM2 confidence score per detection
            scores_np = scores[:, 0] if scores.ndim == 2 else scores

            # 1. Score filter — drop detections below threshold
            keep = [i for i, s in enumerate(scores_np) if float(s) >= self.score_threshold]

            # 2. Overlap filter — when two masks overlap significantly keep the
            #    higher-scoring one (NMS-style over mask IoU)
            filtered: list[int] = []
            for i in keep:
                mask_i = masks[i].astype(bool)
                dominated = False
                for j in filtered:
                    mask_j = masks[j].astype(bool)
                    inter = (mask_i & mask_j).sum()
                    union = (mask_i | mask_j).sum()
                    iou = inter / union if union > 0 else 0.0
                    if iou >= self.overlap_iou_threshold:
                        # Keep whichever has the higher score
                        if float(scores_np[i]) > float(scores_np[j]):
                            filtered.remove(j)
                        else:
                            dominated = True
                        break
                if not dominated:
                    filtered.append(i)

            kept_masks  = masks[filtered]
            kept_scores = scores_np[filtered]
            kept_boxes  = boxes[filtered] if hasattr(boxes, '__getitem__') else torch.stack([boxes[i] for i in filtered])
            kept_labels = [labels[i] for i in filtered]

            mask_dict = MaskDictionaryModel(
                promote_type="mask", mask_name=f"mask_{self.total_frames:05d}.npy"
            )
            mask_list = torch.tensor(kept_masks).to(self.device)
            mask_dict.add_new_frame_annotation(
                mask_list=mask_list,
                box_list=torch.tensor(kept_boxes) if not isinstance(kept_boxes, torch.Tensor) else kept_boxes,
                label_list=kept_labels,
            )
            # Store SAM2 scores into logit field for downstream use
            for idx, obj_info in enumerate(mask_dict.labels.values()):
                obj_info.logit = float(kept_scores[idx])

            self.objects_count = mask_dict.update_masks(
                tracking_annotation_dict=self.last_mask_dict,
                iou_threshold=0.3,
                objects_count=self.objects_count,
            )

            frame_idx = self._add_new_frame(image_np)
            self.video_predictor.reset_state(self.inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, _, _ = self.video_predictor.add_new_mask(
                    self.inference_state, frame_idx, object_id, object_info.mask,
                )

            # Consolidate temp outputs → cond_frame_outputs and run memory encoder
            # so that _run_single_frame_inference can access the conditioning memory.
            self.video_predictor.propagate_in_video_preflight(self.inference_state)

            self.track_dict = copy.deepcopy(mask_dict)
            self.last_mask_dict = mask_dict

        else:
            frame_idx = self._add_new_frame(image_np)

        frame_idx, obj_ids, video_res_masks = self._infer_single_frame(frame_idx)

        frame_masks = MaskDictionaryModel()
        for i, obj_id in enumerate(obj_ids):
            out_mask = video_res_masks[i] > 0.0
            object_info = ObjectInfo(
                instance_id=obj_id,
                mask=out_mask[0],
                class_name=self.track_dict.get_target_class_name(obj_id),
                logit=self.track_dict.get_target_logit(obj_id),
            )
            object_info.update_box()
            frame_masks.labels[obj_id] = object_info
            frame_masks.mask_name = f"mask_{frame_idx:05d}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        self.last_mask_dict = copy.deepcopy(frame_masks)

        H, W = image_np.shape[:2]
        mask_img = torch.zeros((H, W), dtype=torch.int32)
        for obj_id, obj_info in self.last_mask_dict.labels.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_array = mask_img.cpu().numpy()
        annotated_frame = self.visualize_frame_with_mask_and_metadata(
            image_np=image_np,
            mask_array=mask_array,
            json_metadata=self.last_mask_dict.to_dict(),
        )

        self.total_frames += 1
        torch.cuda.empty_cache()
        return annotated_frame

    def set_prompt(self, new_prompt: str):
        self.prompt_text = new_prompt
        self.total_frames = 0
        self.inference_state = self._make_inference_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        print(f"[Prompt Updated] '{new_prompt}'. Tracker state reset.")

    def save_current_state(self, output_dir, raw_image: np.ndarray = None):
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        image_data_dir = os.path.join(output_dir, "images")
        vis_data_dir = os.path.join(output_dir, "result")

        for d in (mask_data_dir, json_data_dir, image_data_dir, vis_data_dir):
            os.makedirs(d, exist_ok=True)

        frame_masks = self.last_mask_dict
        if not frame_masks.mask_name or not frame_masks.mask_name.endswith(".npy"):
            frame_masks.mask_name = f"mask_{self.total_frames:05d}.npy"

        base_name = f"image_{self.total_frames:05d}"

        mask_img = torch.zeros(frame_masks.mask_height, frame_masks.mask_width)
        for obj_id, obj_info in frame_masks.labels.items():
            mask_img[obj_info.mask == True] = obj_id
        np.save(os.path.join(mask_data_dir, frame_masks.mask_name), mask_img.numpy().astype(np.uint16))

        json_path = os.path.join(json_data_dir, base_name + ".json")
        frame_masks.to_json(json_path)

        if raw_image is not None:
            image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image_data_dir, base_name + ".jpg"), image_bgr)

            annotated_image = self.visualize_frame_with_mask_and_metadata(
                image_np=raw_image,
                mask_array=mask_img.numpy().astype(np.uint16),
                json_metadata=frame_masks.to_dict(),
            )
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_data_dir, base_name + "_annotated.jpg"), annotated_bgr)

    def visualize_frame_with_mask_and_metadata(self, image_np, mask_array, json_metadata):
        image = image_np.copy()
        metadata_lookup = json_metadata.get("labels", {})

        all_object_ids, all_object_boxes, all_object_classes, all_object_masks = [], [], [], []

        for obj_id_str, obj_info in metadata_lookup.items():
            instance_id = obj_info.get("instance_id")
            if instance_id is None or instance_id == 0:
                continue
            if instance_id not in np.unique(mask_array):
                continue
            object_mask = mask_array == instance_id
            all_object_ids.append(instance_id)
            all_object_boxes.append([obj_info.get("x1", 0), obj_info.get("y1", 0),
                                     obj_info.get("x2", 0), obj_info.get("y2", 0)])
            all_object_classes.append(obj_info.get("class_name", "unknown"))
            all_object_masks.append(object_mask[None])

        if not all_object_ids:
            return image

        paired = sorted(zip(all_object_ids, all_object_boxes, all_object_masks, all_object_classes))
        all_object_ids    = [p[0] for p in paired]
        all_object_boxes  = [p[1] for p in paired]
        all_object_masks  = [p[2] for p in paired]
        all_object_classes = [p[3] for p in paired]

        all_object_masks = np.concatenate(all_object_masks, axis=0)
        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )
        labels = [f"{iid}: {cls}" for iid, cls in zip(all_object_ids, all_object_classes)]

        annotated = image.copy()
        annotated = sv.MaskAnnotator().annotate(annotated, detections)
        annotated = sv.BoxAnnotator().annotate(annotated, detections)
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
        return annotated

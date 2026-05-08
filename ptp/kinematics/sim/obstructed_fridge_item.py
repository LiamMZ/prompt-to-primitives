from robocasa.environments.kitchen.kitchen import *
import robocasa.utils.object_utils as OU


class ObstructedFridgeItem(Kitchen):
    """
    Obstructed Fridge Item: 

    Simulates a cluttered manipulation problem where the agent needs to retrieve 
    a specific target item from the fridge, but it is obstructed by several 
    other objects in the front.

    Steps:
        Move or circumvent the obstructing objects to grasp the target item,
        and place it onto the counter.
    """

    # certain side-by-side fridges only have 1 shelf, but should be fine if we use top shelf
    EXCLUDE_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(
        self, obj_registries=("aigen", "objaverse", "lightwheel"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )

        if "refs" in self._ep_meta:
            self.rack_index = self._ep_meta["refs"]["rack_index"]
        else:
            self.rack_index = -1 if self.rng.random() < 0.5 else -2

        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        target_name = self.get_obj_lang("target_item")

        ep_meta["lang"] = (
            f"Navigate through the clutter to get the {target_name} from the fridge and place it on the counter."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["rack_index"] = self.rack_index
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # Target item in the back of the fridge shelf
        cfgs.append(
            dict(
                name="target_item",
                obj_groups="all",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.3, 0.2), # w, d
                    pos=(0.0, 1.0), # Back of the shelf
                ),
            )
        )

        # Obstructing items in the front/middle
        cfgs.append(
            dict(
                name="distractor_1",
                obj_groups="all",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.3, 0.2),
                    pos=(-0.25, -0.2), # In front and slightly left
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_2",
                obj_groups="all",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.3, 0.2),
                    pos=(0.0, -0.5), # In front center
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_3",
                obj_groups="all",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.3, 0.2),
                    pos=(0.25, -0.2), # In front and slightly right
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_4",
                obj_groups="all",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.3, 0.2),
                    pos=(0.3, 0.8), # To the right of the target
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_5",
                obj_groups="all",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.rack_index,
                    ),
                    size=(0.3, 0.2),
                    pos=(-0.3, 0.8), # To the left of the target
                ),
            )
        )

        return cfgs

    def _check_success(self):
        target_on_counter = self.check_contact(
            self.objects["target_item"], self.counter
        )

        gripper_far = OU.gripper_obj_far(self, "target_item")

        return target_on_counter and gripper_far

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class LoaderSpec:
    dataset_path_attr: str
    data_list_attr: str
    batch_size: int = 1
    num_workers: int = 4
    split: str = "test"
    suffix: str = ".tif"
    crop_size_attr: str = "crop_size"
    fixed_crop_size: Optional[int] = None

    def resolve(self, args):
        crop_size = self.fixed_crop_size
        if crop_size is None:
            crop_size = getattr(args, self.crop_size_attr)
        return {
            "dataset_path": getattr(args, self.dataset_path_attr),
            "data_list": getattr(args, self.data_list_attr),
            "crop_size": crop_size,
            "split": self.split,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "suffix": self.suffix,
        }


@dataclass(frozen=True)
class TaskRuntimeSpec:
    eval_interval: int = 500
    eval_loaders: Dict[str, LoaderSpec] = field(default_factory=dict)
    infer_loader: Optional[LoaderSpec] = None


XBD_DAMAGE_COLOR_MAP = {
    "background": (0, 0, 0),
    "no_damage": (70, 181, 121),
    "minor_damage": (167, 187, 27),
    "major_damage": (228, 189, 139),
    "destroy": (181, 70, 70),
}

XBD_DAMAGE_LABELS = {
    "background": 0,
    "no_damage": 1,
    "minor_damage": 2,
    "major_damage": 3,
    "destroy": 4,
}

SECOND_COLOR_MAP = {
    "background": (0, 0, 0),
    "low_vegetation": (0, 128, 0),
    "nvg_surface": (128, 128, 128),
    "tree": (0, 255, 0),
    "water": (0, 0, 255),
    "Building": (128, 0, 0),
    "Playground": (255, 0, 0),
}

SECOND_LABELS = {
    "background": 0,
    "low_vegetation": 1,
    "nvg_surface": 2,
    "tree": 3,
    "water": 4,
    "Building": 5,
    "Playground": 6,
}

# BRIGHT shares the same damage color semantics as xBD.
BRIGHT_DAMAGE_COLOR_MAP = XBD_DAMAGE_COLOR_MAP
BRIGHT_DAMAGE_LABELS = XBD_DAMAGE_LABELS


TASK_RUNTIME_SPECS = {
    "bcd": TaskRuntimeSpec(
        eval_interval=500,
        eval_loaders={
            "Validation": LoaderSpec(
                dataset_path_attr="test_dataset_path",
                data_list_attr="test_data_name_list",
            )
        },
        infer_loader=LoaderSpec(
            dataset_path_attr="test_dataset_path",
            data_list_attr="test_data_name_list",
        ),
    ),
    "bda": TaskRuntimeSpec(
        eval_interval=750,
        eval_loaders={
            "Validation": LoaderSpec(
                dataset_path_attr="test_dataset_path",
                data_list_attr="test_data_name_list",
            )
        },
        infer_loader=LoaderSpec(
            dataset_path_attr="test_dataset_path",
            data_list_attr="test_data_name_list",
        ),
    ),
    "bright": TaskRuntimeSpec(
        eval_interval=500,
        eval_loaders={
            "Validation": LoaderSpec(
                dataset_path_attr="val_dataset_path",
                data_list_attr="val_data_name_list",
                batch_size=4,
                num_workers=1,
            ),
            "Test": LoaderSpec(
                dataset_path_attr="test_dataset_path",
                data_list_attr="test_data_name_list",
                batch_size=4,
                num_workers=1,
            ),
        },
    ),
    "scd": TaskRuntimeSpec(
        eval_interval=500,
        eval_loaders={
            "Validation": LoaderSpec(
                dataset_path_attr="test_dataset_path",
                data_list_attr="test_data_name_list",
            )
        },
        infer_loader=LoaderSpec(
            dataset_path_attr="test_dataset_path",
            data_list_attr="test_data_name_list",
            fixed_crop_size=256,
        ),
    ),
}


def get_task_runtime_spec(task_name):
    return TASK_RUNTIME_SPECS.get(task_name)

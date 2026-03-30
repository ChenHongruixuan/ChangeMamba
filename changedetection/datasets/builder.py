from torch.utils.data import DataLoader

from .change_detection import ChangeDetectionDataset
from .common import img_loader
from .damage_assessment import DamageAssessmentDataset
from .multimodal_damage_assessment import MultimodalDamageAssessmentDataset
from .semantic_change_detection import SemanticChangeDetectionDataset


def _is_cd_dataset(dataset_name):
    return any(name in dataset_name for name in ("SYSU", "LEVIR-CD+", "WHU"))


def build_dataset(
    dataset_name,
    *,
    dataset_path,
    data_list,
    crop_size,
    max_iters=None,
    batch_size=1,
    split="train",
    suffix=".tif",
    data_loader=img_loader,
):
    if _is_cd_dataset(dataset_name):
        return ChangeDetectionDataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            data_loader=data_loader,
        )
    if "xBD" in dataset_name:
        return DamageAssessmentDataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            data_loader=data_loader,
        )
    if "SECOND" in dataset_name:
        return SemanticChangeDetectionDataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            data_loader=data_loader,
        )
    if "BRIGHT" in dataset_name:
        return MultimodalDamageAssessmentDataset(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            suffix=suffix,
            data_loader=data_loader,
        )
    raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


def resolve_train_batch_size(args):
    return args.train_batch_size if "BRIGHT" in args.dataset else args.batch_size


def resolve_train_num_workers(args):
    if "BRIGHT" in args.dataset:
        return 4 if args.num_workers is None else args.num_workers
    if "xBD" in args.dataset:
        return 6
    if "SECOND" in args.dataset or _is_cd_dataset(args.dataset):
        return 16
    raise NotImplementedError(f"Unsupported dataset: {args.dataset}")


def build_train_loader(args, **kwargs):
    batch_size = resolve_train_batch_size(args)
    dataset = build_dataset(
        args.dataset,
        dataset_path=args.train_dataset_path,
        data_list=args.train_data_name_list,
        crop_size=args.crop_size,
        max_iters=args.max_iters,
        batch_size=batch_size,
        split=args.type,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=resolve_train_num_workers(args),
        drop_last=False,
        **kwargs,
    )


def build_eval_loader(
    dataset_name,
    *,
    dataset_path,
    data_list,
    crop_size,
    split="test",
    batch_size=1,
    num_workers=4,
    suffix=".tif",
    shuffle=False,
    **kwargs,
):
    dataset = build_dataset(
        dataset_name,
        dataset_path=dataset_path,
        data_list=data_list,
        crop_size=crop_size,
        batch_size=batch_size,
        split=split,
        suffix=suffix,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        **kwargs,
    )


def make_data_loader(args, **kwargs):
    return build_train_loader(args, **kwargs)

import numpy as np

from changedetection.checkpoints import load_model_weights


def get_vssm_kwargs(config):
    """Extract all VSSM backbone parameters from config into a dict.

    Usage:
        model = ChangeMambaBCD(pretrained=args.pretrained_weight_path, **get_vssm_kwargs(config))
    """
    return dict(
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )


def load_checkpoint(model, path):
    """Load model weights from either a raw state dict or a full training checkpoint."""
    return load_model_weights(model, path)


def read_name_list(path):
    with open(path, "r") as handle:
        return [data_name.strip() for data_name in handle]


def populate_name_lists(args, mapping):
    for path_attr, list_attr in mapping.items():
        setattr(args, list_attr, read_name_list(getattr(args, path_attr)))


def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    """Map an integer label array to an RGB color image.

    Args:
        labels: 2-D numpy array of integer class indices (H, W).
        ori_label_value_dict: dict mapping class name -> RGB tuple, e.g. {'background': (0,0,0)}.
        target_label_value_dict: dict mapping class name -> integer index.

    Returns:
        color_mapped: uint8 numpy array of shape (H, W, 3).
    """
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    H, W = labels.shape
    color_mapped = np.zeros((H, W, 3), dtype=np.uint8)
    for target_label, ori_label in target_to_ori.items():
        color_mapped[labels == target_label] = ori_label_value_dict[ori_label]
    return color_mapped

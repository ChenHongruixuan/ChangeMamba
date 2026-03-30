import os
from collections.abc import Mapping

import torch

from changedetection.logging_utils import format_log_block


def _read_checkpoint(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"No checkpoint found at '{path}'")
    return torch.load(path, map_location="cpu")


def _strip_prefix_if_present(state_dict, prefix):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _normalize_state_dict(state_dict):
    normalized = dict(state_dict)
    for prefix in ("module.", "model."):
        normalized = _strip_prefix_if_present(normalized, prefix)
    return normalized


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _normalize_state_dict(value)
    if isinstance(checkpoint, Mapping):
        return _normalize_state_dict(checkpoint)
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def _match_state_dict(model, checkpoint_state_dict):
    model_state = model.state_dict()
    matched = {}
    unexpected = []
    mismatched = []

    for key, value in checkpoint_state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if getattr(model_state[key], "shape", None) != getattr(value, "shape", None):
            mismatched.append(
                {
                    "key": key,
                    "model_shape": tuple(model_state[key].shape),
                    "checkpoint_shape": tuple(value.shape),
                }
            )
            continue
        matched[key] = value

    merged_state = dict(model_state)
    merged_state.update(matched)
    model.load_state_dict(merged_state)
    return {
        "loaded_keys": len(matched),
        "missing_keys": sorted(set(model_state) - set(matched)),
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
    }


def load_model_weights(model, path):
    checkpoint = _read_checkpoint(path)
    load_info = _match_state_dict(model, extract_model_state_dict(checkpoint))
    load_info["path"] = path
    return load_info


def _preview_sequence(items, max_items=8):
    items = list(items)
    if not items:
        return "[]"
    preview_items = items[:max_items]
    if len(items) > max_items:
        preview_items.append(f"... (+{len(items) - max_items} more)")
    return preview_items


def _preview_mismatched_keys(items, max_items=6):
    if not items:
        return "[]"
    preview_items = []
    for item in items[:max_items]:
        preview_items.append(
            f"{item['key']} (ckpt={item['checkpoint_shape']}, model={item['model_shape']})"
        )
    if len(items) > max_items:
        preview_items.append(f"... (+{len(items) - max_items} more)")
    return preview_items


def format_checkpoint_load_report(load_info, title="CHECKPOINT Load"):
    values = {
        "matched": load_info["loaded_keys"],
        "missing": len(load_info["missing_keys"]),
        "unexpected": len(load_info["unexpected_keys"]),
        "mismatched": len(load_info["mismatched_keys"]),
    }
    if load_info["missing_keys"]:
        values["missing_preview"] = _preview_sequence(load_info["missing_keys"])
    if load_info["unexpected_keys"]:
        values["unexpected_preview"] = _preview_sequence(load_info["unexpected_keys"])
    if load_info["mismatched_keys"]:
        values["mismatched_preview"] = _preview_mismatched_keys(load_info["mismatched_keys"])
    return format_log_block(
        title,
        values,
        meta={"path": load_info["path"]},
    )


def _safe_load_component(component, state_dict, component_name):
    if component is None or state_dict is None:
        return False, None
    try:
        component.load_state_dict(state_dict)
        return True, None
    except Exception as exc:  # pragma: no cover - defensive compatibility path
        return False, f"Failed to load {component_name} state: {exc}"


def resume_training_state(path, *, model, optimizer=None, scheduler=None):
    checkpoint = _read_checkpoint(path)
    load_info = _match_state_dict(model, extract_model_state_dict(checkpoint))

    optimizer_loaded, optimizer_error = _safe_load_component(
        optimizer,
        checkpoint.get("optimizer") if isinstance(checkpoint, Mapping) else None,
        "optimizer",
    )
    scheduler_loaded, scheduler_error = _safe_load_component(
        scheduler,
        checkpoint.get("scheduler") if isinstance(checkpoint, Mapping) else None,
        "scheduler",
    )

    if not isinstance(checkpoint, Mapping):
        checkpoint = {}

    return {
        "path": path,
        "iteration": int(checkpoint.get("iteration", checkpoint.get("iter", 0)) or 0),
        "best_score": checkpoint.get("best_score"),
        "best_record": checkpoint.get("best_record"),
        "task_name": checkpoint.get("task_name"),
        "config_dump": checkpoint.get("config_dump"),
        "args_snapshot": checkpoint.get("args_snapshot"),
        "extra_state": checkpoint.get("extra_state", {}),
        "optimizer_loaded": optimizer_loaded,
        "optimizer_error": optimizer_error,
        "scheduler_loaded": scheduler_loaded,
        "scheduler_error": scheduler_error,
        "load_info": load_info,
    }


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    iteration=0,
    best_score=None,
    best_record=None,
    task_name=None,
    config=None,
    args=None,
    extra_state=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": getattr(model, "module", model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "iteration": int(iteration),
        "best_score": best_score,
        "best_record": best_record,
        "task_name": task_name,
        "config_dump": config.dump() if config is not None and hasattr(config, "dump") else None,
        "args_snapshot": vars(args) if args is not None else None,
        "extra_state": extra_state or {},
    }
    torch.save(payload, path)

from importlib import import_module


TRAINERS = {
    "bcd": ("changedetection.tasks.bcd", "BCDTrainer"),
    "bda": ("changedetection.tasks.bda", "BDATrainer"),
    "bright": ("changedetection.tasks.bright", "BRIGHTTrainer"),
    "scd": ("changedetection.tasks.scd", "SCDTrainer"),
}

INFERERS = {
    "bcd": ("changedetection.tasks.bcd", "BCDInferer"),
    "bda": ("changedetection.tasks.bda", "BDAInferer"),
    "scd": ("changedetection.tasks.scd", "SCDInferer"),
}


def _load_entry(entry):
    module_name, attr_name = entry
    module = import_module(module_name)
    return getattr(module, attr_name)


def get_trainer(task_name):
    return _load_entry(TRAINERS[task_name])


def get_inferer(task_name):
    return _load_entry(INFERERS[task_name])

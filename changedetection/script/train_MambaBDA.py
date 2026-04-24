import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse

from changedetection.script.script_utils import add_weight_loading_args, normalize_weight_loading_args, populate_name_lists
from changedetection.tasks import get_trainer


def main():
    parser = argparse.ArgumentParser(description="Training on xBD dataset")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    add_weight_loading_args(parser, allow_model_checkpoint=True, allow_resume_training=True)
    parser.add_argument("--dataset", type=str, default="xBD")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--train_data_list_path", type=str)
    parser.add_argument("--test_dataset_path", type=str)
    parser.add_argument("--test_data_list_path", type=str)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--train_data_name_list", type=list)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_iters", type=int, default=80000)
    parser.add_argument("--model_type", type=str, default="ChangeMamba-BDA")
    parser.add_argument("--model_param_path", type=str, default="../saved_models")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    args = parser.parse_args()
    normalize_weight_loading_args(args, parser, mode="train")
    populate_name_lists(
        args,
        {
            "train_data_list_path": "train_data_name_list",
            "test_data_list_path": "test_data_name_list",
        },
    )
    get_trainer("bda")(args).training()


if __name__ == "__main__":
    main()

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse

from changedetection.script.script_utils import populate_name_lists
from changedetection.tasks import get_inferer


def main():
    parser = argparse.ArgumentParser(description="Inference on SECOND dataset")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")
    parser.add_argument("--pretrained_weight_path", type=str)
    parser.add_argument("--dataset", type=str, default="SECOND")
    parser.add_argument("--test_dataset_path", type=str)
    parser.add_argument("--test_data_list_path", type=str)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="MambaSCD_Tiny")
    parser.add_argument("--result_saved_path", type=str, default="../results")
    parser.add_argument("--resume", type=str)

    args = parser.parse_args()
    populate_name_lists(args, {"test_data_list_path": "test_data_name_list"})
    get_inferer("scd")(args).infer()


if __name__ == "__main__":
    main()

import imageio
import numpy as np
from torch.utils.data import Dataset


def img_loader(path):
    return np.array(imageio.imread(path), np.float32)


def expand_data_list(data_list, max_iters=None, batch_size=1):
    if max_iters is None:
        return list(data_list)

    num_samples = max_iters * batch_size
    expanded = list(data_list) * int(np.ceil(float(num_samples) / len(data_list)))
    return expanded[:num_samples]


class BaseChangeDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        batch_size=1,
        split="train",
        data_loader=img_loader,
    ):
        self.dataset_path = dataset_path
        self.data_list = expand_data_list(data_list, max_iters=max_iters, batch_size=batch_size)
        self.crop_size = crop_size
        self.split = split
        self.is_train = "train" in split
        self.loader = data_loader

    def __len__(self):
        return len(self.data_list)

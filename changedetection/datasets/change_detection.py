import os

import numpy as np

from . import imutils
from .common import BaseChangeDataset


class ChangeDetectionDataset(BaseChangeDataset):
    def _transform(self, pre_img, post_img, label):
        if self.is_train:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.to_channel_first(imutils.normalize_img(pre_img))
        post_img = imutils.to_channel_first(imutils.normalize_img(post_img))
        return pre_img, post_img, np.asarray(label)

    def __getitem__(self, index):
        item_name = self.data_list[index]
        pre_path = os.path.join(self.dataset_path, "T1", item_name)
        post_path = os.path.join(self.dataset_path, "T2", item_name)
        label_path = os.path.join(self.dataset_path, "GT", item_name)

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path) / 255

        pre_img, post_img, label = self._transform(pre_img, post_img, label)
        return pre_img, post_img, label, item_name

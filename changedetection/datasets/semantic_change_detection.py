import os

import numpy as np

from . import imutils
from .common import BaseChangeDataset


class SemanticChangeDetectionDataset(BaseChangeDataset):
    def _resolve_paths(self, item_name):
        suffix = ".png" if self.is_train else ""
        return (
            os.path.join(self.dataset_path, "T1", item_name + suffix),
            os.path.join(self.dataset_path, "T2", item_name + suffix),
            os.path.join(self.dataset_path, "GT_T1", item_name + suffix),
            os.path.join(self.dataset_path, "GT_T2", item_name + suffix),
            os.path.join(self.dataset_path, "GT_CD", item_name + suffix),
        )

    def _transform(self, pre_img, post_img, cd_label, t1_label, t2_label):
        if self.is_train:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size
            )
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label
            )
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label
            )
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label
            )

        pre_img = imutils.to_channel_first(imutils.normalize_img(pre_img))
        post_img = imutils.to_channel_first(imutils.normalize_img(post_img))
        return pre_img, post_img, np.asarray(cd_label), np.asarray(t1_label), np.asarray(t2_label)

    def __getitem__(self, index):
        item_name = self.data_list[index]
        pre_path, post_path, t1_label_path, t2_label_path, cd_label_path = self._resolve_paths(item_name)

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(t1_label_path)
        t2_label = self.loader(t2_label_path)
        cd_label = self.loader(cd_label_path) / 255

        pre_img, post_img, cd_label, t1_label, t2_label = self._transform(
            pre_img, post_img, cd_label, t1_label, t2_label
        )
        return pre_img, post_img, cd_label, t1_label, t2_label, item_name

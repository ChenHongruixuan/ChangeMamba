import os

import numpy as np

from . import imutils
from .common import BaseChangeDataset, img_loader


class MultimodalDamageAssessmentDataset(BaseChangeDataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        batch_size=1,
        split="train",
        data_loader=img_loader,
        suffix=".tif",
    ):
        super().__init__(
            dataset_path=dataset_path,
            data_list=data_list,
            crop_size=crop_size,
            max_iters=max_iters,
            batch_size=batch_size,
            split=split,
            data_loader=data_loader,
        )
        self.suffix = suffix

    def _transform(self, pre_img, post_img, clf_label):
        if self.is_train:
            pre_img, post_img, clf_label = imutils.random_crop_new(pre_img, post_img, clf_label, self.crop_size)
            pre_img, post_img, clf_label = imutils.random_fliplr(pre_img, post_img, clf_label)
            pre_img, post_img, clf_label = imutils.random_flipud(pre_img, post_img, clf_label)
            pre_img, post_img, clf_label = imutils.random_rot(pre_img, post_img, clf_label)

        pre_img = imutils.to_channel_first(imutils.normalize_img(pre_img))
        post_img = imutils.to_channel_first(imutils.normalize_img(post_img))
        return pre_img, post_img, np.asarray(clf_label)

    def __getitem__(self, index):
        item_name = self.data_list[index]
        pre_path = os.path.join(self.dataset_path, "pre-event", f"{item_name}_pre_disaster{self.suffix}")
        post_path = os.path.join(self.dataset_path, "post-event", f"{item_name}_post_disaster{self.suffix}")
        label_path = os.path.join(self.dataset_path, "target", f"{item_name}_building_damage{self.suffix}")

        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = np.stack((self.loader(post_path),) * 3, axis=-1)
        clf_label = self.loader(label_path)

        pre_img, post_img, clf_label = self._transform(pre_img, post_img, clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1
        return pre_img, post_img, loc_label, clf_label, item_name

import os

import numpy as np

from . import imutils
from .common import BaseChangeDataset


class DamageAssessmentDataset(BaseChangeDataset):
    def _resolve_paths(self, item_name):
        if self.is_train:
            parts = item_name.rsplit("_", 2)
            pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"
        else:
            pre_img_name = f"{item_name}_pre_disaster.png"
            post_img_name = f"{item_name}_post_disaster.png"

        return (
            os.path.join(self.dataset_path, "images", pre_img_name),
            os.path.join(self.dataset_path, "images", post_img_name),
            os.path.join(self.dataset_path, "masks", pre_img_name),
            os.path.join(self.dataset_path, "masks", post_img_name),
        )

    def _transform(self, pre_img, post_img, loc_label, clf_label):
        if self.is_train:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(
                pre_img, post_img, loc_label, clf_label, self.crop_size
            )
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(
                pre_img, post_img, loc_label, clf_label
            )
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(
                pre_img, post_img, loc_label, clf_label
            )
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(
                pre_img, post_img, loc_label, clf_label
            )
            clf_label = clf_label.copy()
            clf_label[clf_label == 0] = 255

        pre_img = imutils.to_channel_first(imutils.normalize_img(pre_img))
        post_img = imutils.to_channel_first(imutils.normalize_img(post_img))
        return pre_img, post_img, np.asarray(loc_label), np.asarray(clf_label)

    def __getitem__(self, index):
        item_name = self.data_list[index]
        pre_path, post_path, loc_label_path, clf_label_path = self._resolve_paths(item_name)

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)[:, :, 0]
        clf_label = self.loader(clf_label_path)[:, :, 0]

        pre_img, post_img, loc_label, clf_label = self._transform(pre_img, post_img, loc_label, clf_label)
        return pre_img, post_img, loc_label, clf_label, item_name

import random

import numpy as np


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]

    return normalized_img


def to_channel_first(img):
    return np.transpose(img, (2, 0, 1))


def _pad_array(array, target_h, target_w, offset_h, offset_w, fill_value):
    if array.ndim == 3:
        padded = np.empty((target_h, target_w, array.shape[2]), dtype=np.float32)
        padded[...] = np.asarray(fill_value, dtype=np.float32)
        padded[offset_h:(offset_h + array.shape[0]), offset_w:(offset_w + array.shape[1]), :] = array
        return padded

    padded = np.full((target_h, target_w), fill_value, dtype=np.float32)
    padded[offset_h:(offset_h + array.shape[0]), offset_w:(offset_w + array.shape[1])] = array
    return padded


def _sample_crop_box(label, crop_size, ignore_index=255, cat_max_ratio=0.75):
    height, width = label.shape
    for _ in range(10):
        h_start = random.randrange(0, height - crop_size + 1, 1)
        h_end = h_start + crop_size
        w_start = random.randrange(0, width - crop_size + 1, 1)
        w_end = w_start + crop_size

        temp_label = label[h_start:h_end, w_start:w_end]
        classes, counts = np.unique(temp_label, return_counts=True)
        counts = counts[classes != ignore_index]
        if len(counts) > 1 and np.max(counts) / np.sum(counts) < cat_max_ratio:
            return h_start, h_end, w_start, w_end

    h_start = random.randrange(0, height - crop_size + 1, 1)
    h_end = h_start + crop_size
    w_start = random.randrange(0, width - crop_size + 1, 1)
    w_end = w_start + crop_size
    return h_start, h_end, w_start, w_end


def random_crop_multi(
    images,
    labels,
    crop_size,
    *,
    selection_label=0,
    mean_rgb=(0, 0, 0),
    ignore_index=255,
):
    reference = labels[selection_label]
    h, w = reference.shape
    target_h = max(crop_size, h)
    target_w = max(crop_size, w)

    h_pad = int(np.random.randint(target_h - h + 1))
    w_pad = int(np.random.randint(target_w - w + 1))

    padded_images = [
        _pad_array(image, target_h, target_w, h_pad, w_pad, mean_rgb)
        for image in images
    ]
    padded_labels = [
        _pad_array(label, target_h, target_w, h_pad, w_pad, ignore_index)
        for label in labels
    ]

    h_start, h_end, w_start, w_end = _sample_crop_box(
        padded_labels[selection_label],
        crop_size,
        ignore_index=ignore_index,
    )

    cropped_images = [image[h_start:h_end, w_start:w_end, :] for image in padded_images]
    cropped_labels = [label[h_start:h_end, w_start:w_end] for label in padded_labels]
    return (*cropped_images, *cropped_labels)


def _random_apply(transform, *arrays):
    return tuple(transform(array) for array in arrays)


def _random_rotate(*arrays):
    k = random.randrange(3) + 1
    return tuple(np.rot90(array, k).copy() for array in arrays)


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        return _random_apply(np.fliplr, pre_img, post_img, label)
    return pre_img, post_img, label


def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        return _random_apply(np.fliplr, pre_img, post_img, label_1, label_2)
    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        return _random_apply(np.fliplr, pre_img, post_img, label_cd, label_1, label_2)
    return pre_img, post_img, label_cd, label_1, label_2


def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        return _random_apply(np.flipud, pre_img, post_img, label)
    return pre_img, post_img, label


def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        return _random_apply(np.flipud, pre_img, post_img, label_1, label_2)
    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        return _random_apply(np.flipud, pre_img, post_img, label_cd, label_1, label_2)
    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    return _random_rotate(pre_img, post_img, label)


def random_rot_bda(pre_img, post_img, label_1, label_2):
    return _random_rotate(pre_img, post_img, label_1, label_2)


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    return _random_rotate(pre_img, post_img, label_cd, label_1, label_2)


def random_crop(img, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    cropped_img, = random_crop_multi(
        [img],
        [np.zeros(img.shape[:2], dtype=np.float32)],
        crop_size,
        selection_label=0,
        mean_rgb=mean_rgb,
        ignore_index=ignore_index,
    )
    return cropped_img


def random_bi_image_crop(pre_img, obj, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    cropped_pre_img, cropped_obj = random_crop_multi(
        [pre_img],
        [obj],
        crop_size,
        selection_label=0,
        mean_rgb=mean_rgb,
        ignore_index=ignore_index,
    )
    return cropped_pre_img, cropped_obj


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    return random_crop_multi(
        [pre_img, post_img],
        [label],
        crop_size,
        selection_label=0,
        mean_rgb=mean_rgb,
        ignore_index=ignore_index,
    )


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    return random_crop_multi(
        [pre_img, post_img],
        [loc_label, clf_label],
        crop_size,
        selection_label=0,
        mean_rgb=mean_rgb,
        ignore_index=ignore_index,
    )


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    return random_crop_multi(
        [pre_img, post_img],
        [label_cd, label_1, label_2],
        crop_size,
        selection_label=1,
        mean_rgb=mean_rgb,
        ignore_index=ignore_index,
    )

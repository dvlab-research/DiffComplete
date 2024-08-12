import numpy as np
from PIL import Image, ImageDraw
import math
import random
import argparse
import os
from utils.visualize_voxel import visualize_data

def toU8(sample):
    if sample is None:
        return sample

    sample = np.clip(((sample + 1) * 127.5), 0, 255).astype(np.uint8)
    sample = np.transpose(sample, (1, 2, 0))
    return sample

def write_images(voxels, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    for image_name, voxel in zip(img_names, voxels):
        out_path = os.path.join(dir_path, image_name)
        visualize_data(voxel, 'voxels', out_path)

def RandomCrop(s, tries, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)

    while True:
        mask = np.ones((s, s, s), np.uint8)
        def Fill(max_size):
            # import ipdb; ipdb.set_trace()
            l, w, h = np.random.randint(max_size), np.random.randint(max_size), np.random.randint(max_size)
            ll, ww, hh = l // 2, w // 2, h // 2
            x, y, z = np.random.randint(-ll, s - l + ll), np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(z, 0): min(z + h, s), max(y, 0): min(y + w, s), max(x, 0): min(x + l, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(max_tries):
                Fill(max_size)
        MultiFill(int(tries * coef), s)
        hole_ratio = 1 - np.mean(mask)
        assert hole_ratio >= hole_range[0] and hole_ratio <= hole_range[1]
        return mask[np.newaxis, ...].astype(np.float32)


def HalfCrop(s):

    axis = np.random.randint(3)
    mask = np.ones((s, s, s), np.uint8)
    crop = np.random.randint(int(0.3 * s) , int(0.7 * s))
    if axis == 0:
        if np.random.random() > 0.5:
            mask[: crop, ...] = 0
        else:
            mask[crop:, ...] = 0
    elif axis == 1:
        if np.random.random() > 0.5:
            mask[:, :crop, :] = 0
        else:
            mask[:, crop:, :] = 0
    else:
        if np.random.random() > 0.5:
            mask[..., :crop] = 0
        else:
            mask[..., :crop] = 0

    return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, required=False, default=32)
    parser.add_argument('--tries', type=int, required=False, default=8)
    parser.add_argument('--type', type=str, required=False, default="random")
    args = parser.parse_args()

    cnt = 50
    tot = 0
    dir_path = "./output_mask_half" if args.type == "half" else "./output_mask_random"

    masks = []
    names = []
    for i in range(cnt):
        mask = HalfCrop(s=args.res) if args.type == "half" else RandomCrop(s=args.res, tries=args.tries)
        mask = np.squeeze(mask)
        tot += mask.mean()
        masks.append(mask)
        names.append(f"{i}.jpg")
    print(tot / cnt)
    write_images(masks, names, dir_path)

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data

num_classes = 2
ignore_label = 2
path = 'datasets/QB'
palette = [255, 255, 255, 0, 0, 0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset(mode, root):
    rootAndMode = Path(root, mode)
    input_dir_1 = 'input1'
    input_dir_2 = 'input2'
    label_dir = 'label'

    items = []

    for regionOrPatch in Path(rootAndMode, input_dir_1).iterdir():
        if regionOrPatch.is_dir():
            for patch in regionOrPatch.iterdir():
                items.append((
                    patch,
                    Path(rootAndMode, input_dir_2, regionOrPatch.stem, patch.name),
                    Path(rootAndMode, label_dir, regionOrPatch.stem, patch.name),
                    patch.name
                ))
        else:
            items.append((
                regionOrPatch,
                Path(rootAndMode, input_dir_2, regionOrPatch.stem),
                Path(rootAndMode, label_dir, regionOrPatch.stem),
                regionOrPatch.stem
            ))

    print(items[0])
    return items


class QB(data.Dataset):
    def __init__(self, quality, mode, data_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, subset=False):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        # d_t = np.load('data/camvid_al_splits.npy', allow_pickle=True).item()['d_t']
        #
        # if subset:
        #     self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in d_t)]

        print('Using ' + str(len(self.imgs)) + ' images.')

    def __getitem__(self, index):
        img_path_1, img_path_2, mask_path, im_name = self.imgs[index]
        img1 = Image.open(img_path_1).convert('RGB')
        img2 = Image.open(img_path_2).convert('RGB')
        img = np.concatenate((img1, img2), axis=-1)
        print(img.shape)
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, (img_path_1, img_path_2, mask_path, im_name)

    def __len__(self):
        return len(self.imgs)

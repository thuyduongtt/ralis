import numpy as np
from pathlib import Path


# CamVid:
# Total: 708
# Train, Val, Test: 370, 104, 234
# Train ==> 110 labeled, 260 for d_v
# 110 labeled ==> 10 for d_s, 100 for d_t
# d_r is val set
# Image size 480x360, split into 6x4 = 24 regions, 80x90 each
# K = 24 (regions per step)
#####
# d_s: 10
# d_t: 100
# d_r: 104
# d_v: 260

# Cityscapes:
# Total: 3475
# Train, Val, Test: 2975, 500, 0
# Train ==> 360 labeled, 2615 for d_v
# 360 labeled ==> 10 for d_s, 150 for d_t, 200 for d_r
# Image size 2048x1024, split into 16x8 = 128 regions, 128x128 each
# K = 256 (regions per step)
#####
# d_s: 10
# d_t: 150
# d_r: 200
# d_v: 2615

# QB
# Total: 1411
# Train, Val, Test: 846, 141, 424
# Train ==> 160 labeled, 686 for d_v
# 160 labeled ==> 10 for d_s, 150 for d_t
# d_r is val set
# Image size 256x256, split into 4x4=16 regions, 64x64 each
# K = 24 (regions per step)
#####
# d_s: 10
# d_t: 150
# d_r: 141
# d_v: 686

# QB - Original split
# Total: 1411
# Train, Val, Test: 1350, 30, 31
# Train ==> 160 labeled, 1190 for d_v
# 160 labeled ==> 10 for d_s, 150 for d_t
# d_r is val set
# Image size 256x256, split into 4x4=16 regions, 64x64 each
# K = 24 (regions per step)
#####
# d_s: 10
# d_t: 150
# d_r: 30
# d_v: 1190


def open_ds(mode='train'):
    rootAndMode = Path('QB', mode)
    input_dir_1 = 'input1'

    items = []

    for regionOrPatch in Path(rootAndMode, input_dir_1).iterdir():
        if regionOrPatch.is_dir():
            for patch in regionOrPatch.iterdir():
                items.append(append_patch_region(patch.name, regionOrPatch.stem))
        else:
            items.append(regionOrPatch.name)

    return np.asarray(items)


# append patch region to distinguish patches like "center.bmp", 'bottom_right.bmp"
def append_patch_region(patch_name, region_name):
    if region_name in patch_name:
        return patch_name
    return region_name + '_' + patch_name


def split():
    train_items = open_ds()
    val_items = open_ds('val')

    # see comment above
    n = train_items.shape[0]
    n_labeled = 160
    n_d_v = n - n_labeled
    n_d_s = 10
    n_d_t = n_labeled - n_d_s
    n_d_r = len(val_items)

    print(f'Total: {n}, labeled: {n_labeled}, d_s: {n_d_s}, d_t: {n_d_t}, d_r: {n_d_r}, d_v: {n_d_v}')

    # Train ==> labeled + d_v
    split_flags = np.asarray([1] * n_labeled + [0] * n_d_v)
    np.random.shuffle(split_flags)

    labeled = train_items[split_flags == 1]
    d_v = train_items[split_flags == 0]

    # make sure there is no overlap
    assert not check_overlap(labeled, d_v)

    # labeled ==> d_s + d_t
    split_flags = np.asarray([1] * n_d_s + [0] * n_d_t)
    np.random.shuffle(split_flags)
    d_s = labeled[split_flags == 1]
    d_t = labeled[split_flags == 0]

    # make sure there is no overlap
    assert not check_overlap(d_s, d_t)

    np.save('qb_al_splits.npy', {
        'd_r': val_items.tolist(),
        'd_s': d_s.tolist(),
        'd_t': d_t.tolist(),
        'd_v': d_v.tolist()
    })


def check_overlap(arr1, arr2):
    return np.any(np.isin(arr1, arr2))


if __name__ == '__main__':
    split()


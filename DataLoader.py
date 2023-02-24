# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

from os.path import join as pjoin
import SimpleITK as sitk
# from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

for idx in range(1,22):

# idx = 1
    root = f'C:\sue\\nantes\Dataset\Raw Datasets\Dataset_{idx}'
    img_path = pjoin(root,f'Dataset_{idx}_Mag.nii.gz')

    image = sitk.ReadImage(img_path)
    # label = sitk.ReadImage(pjoin(lbl_root, pid))

    image_mat = sitk.GetArrayFromImage(image)

    print(f'Dataset_{idx}')
    print(image_mat.shape)
    # print(image.GetDirection())
    # print(image.GetOrigin())
    print(image.GetSpacing())

# The features
# label.SetDirection(image.GetDirection())
# label.SetOrigin(image.GetOrigin())
# label.SetSpacing(image.GetSpacing())

# 3. Re-sample to the same spacing.
# resampler = sitk.ResampleImageFilter()
# resampler.SetOutputDirection(image.GetDirection())
# resampler.SetOutputOrigin(image.GetOrigin())

# old_spacing = image.GetSpacing()
# old_size = image.GetSize()

# new_size = [osz * osp / nsp for osz, nsp, osp in zip(old_size, new_spacing, old_spacing)]
# new_size = list(map(int, new_size))
# for i in range(2):
#     new_size[i] = max(crop_size, new_size[i])  # if smaller than given size
# round_new_spacing = [osp * osz / nsz for osp, nsz, osz in zip(old_spacing, new_size, old_size)]
# # print(old_size, new_size, old_spacing, new_spacing)

# resampler.SetOutputSpacing(round_new_spacing)
# resampler.SetSize(new_size)

# # resampler.SetInterpolator(sitk.sitkLinear)
# resampler.SetInterpolator(sitk.sitkBSpline)
# image = resampler.Execute(image)
# resampler.SetInterpolator(sitk.sitkNearestNeighbor)
# label = resampler.Execute(label)
# print(image.GetSize(), label.GetSize(), image.GetSpacing(), label.GetSpacing())

# Save niift files
# save_img_path = pjoin(img_root, f'{pid}.nii.gz')
# save_lbl_path = pjoin(lbl_root, f'{pid}.nii.gz')
# sitk.WriteImage(image, save_img_path)
# sitk.WriteImage(label, save_lbl_path)
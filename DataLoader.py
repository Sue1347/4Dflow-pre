# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())

from os.path import join as pjoin
import SimpleITK as sitk
import scipy.fftpack as fp
# from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def PCMRA(image_mat,vu_mat,vv_mat,vw_mat,gamma_t=False,gamma_idx=0.6,tFrames=10):
    """
    Input: anatomic matrix, volecities matrix
    gamma_t, calculates gamma value of PCMRA, 
    from 0.2(according to the Bustamante-2018) to gamma_t, in tFrames Frames. 
    """
    if(gamma_t):
        gamma = 0.2
        t,x,y,z = image_mat.shape
        PCMRA = np.zeros_like(image_mat)
        for i in range(t):
            if(t<tFrames):
                gamma = 0.2 + np.sin(t * np.pi / tFrames) * gamma_idx
            PCMRA[i] = image_mat[i]*np.power(np.power(vu_mat[i],2)+np.power(vv_mat[i],2)+np.power(vw_mat[i],2),gamma)
    else:
        
        PCMRA = image_mat*np.power(np.power(vu_mat,2)+np.power(vv_mat,2)+np.power(vw_mat,2),0.2)
    
    return PCMRA

if __name__ == '__main__':
    # Read Through the data, and preprocess them
    for idx in range(1,22):

        # Load the data
        root = f'C:\sue\\nantes\Dataset\Raw Datasets\Dataset_{idx}'
        img_path = pjoin(root,f'Dataset_{idx}_Mag.nii.gz')
        vu_path = pjoin(root,f'Dataset_{idx}_Vap.nii.gz')
        vv_path = pjoin(root,f'Dataset_{idx}_Vfh.nii.gz')
        vw_path = pjoin(root,f'Dataset_{idx}_Vlr.nii.gz')
        # magu_path = pjoin(root,f'Dataset_{idx}_Magu.nii.gz')
        # magv_path = pjoin(root,f'Dataset_{idx}_Magv.nii.gz')
        # magw_path = pjoin(root,f'Dataset_{idx}_Magw.nii.gz')

        image = sitk.ReadImage(img_path)
        vu = sitk.ReadImage(vu_path)
        vv = sitk.ReadImage(vv_path)
        vw = sitk.ReadImage(vw_path)
        # magu = sitk.ReadImage(magu_path)
        # magv = sitk.ReadImage(magv_path)
        # magw = sitk.ReadImage(magw_path)

        image_mat = sitk.GetArrayFromImage(image)
        vu_mat = sitk.GetArrayFromImage(vu)
        vv_mat = sitk.GetArrayFromImage(vv)
        vw_mat = sitk.GetArrayFromImage(vw)
        # magu_mat = sitk.GetArrayFromImage(magu)
        # magv_mat = sitk.GetArrayFromImage(magv)
        # magw_mat = sitk.GetArrayFromImage(magw)

        # Information
        print(f'Dataset_{idx}')
        # print(image_mat.min(),image_mat.max())
        # print('Vu:\n',vu_mat.min(),vu_mat.max())
        # print('Vv:\n',vv_mat.min(),vv_mat.max())
        # print('Vw:\n',vw_mat.min(),vw_mat.max())
        # print(image.GetDirection())
        # print(image.GetOrigin())
        # print(image.GetSpacing())

        # fft_m = fp.fft2(image_mat)
        # fft_magu = fp.fft2(magu_mat)
        # fft_magv = fp.fft2(magv_mat)
        # fft_magw = fp.fft2(magw_mat)

        # cdu_path = pjoin(root,f'Dataset_{idx}_CDu.nii.gz')
        # cdu_pre = fp.ifft2(fft_m - fft_magu).real 
        # print(cdu_pre.shape)
        # cdu = sitk.GetImageFromArray(cdu_pre,isVector=False)

        # Save Original PCMRA images
        pcmra_ori_path = pjoin(root,f'Dataset_{idx}_PCMRA_ori.nii.gz')
        pcmra_pre =  PCMRA(image_mat,vu_mat,vv_mat,vw_mat,True)
        print(pcmra_pre.shape)
        pcmra = sitk.GetImageFromArray(pcmra_pre,isVector=False)
        sitk.WriteImage(pcmra, pcmra_ori_path)

        # Save Time-relavent PCMRA images by using sin function
        pcmra_sin_path = pjoin(root,f'Dataset_{idx}_PCMRA_sin.nii.gz')
        pcmra_pre =  PCMRA(image_mat,vu_mat,vv_mat,vw_mat,False)
        print(pcmra_pre.shape)
        pcmra = sitk.GetImageFromArray(pcmra_pre,isVector=False)
        sitk.WriteImage(pcmra, pcmra_sin_path)


    # The features
    # label.SetDirection(image.GetDirection())
    # label.SetOrigin(image.GetOrigin())
    # label.SetSpacing(image.GetSpacing())

    # Re-sample to the same spacing.
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
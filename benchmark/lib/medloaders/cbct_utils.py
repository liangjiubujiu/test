# -*- coding:utf-8 -*-
import os
import numpy as np
import pydicom
from PIL import Image
import cv2
import pickle
import torch


def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    if not contours:
        return mask
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def convert_from_dicom_to_jpg(img, low_window, high_window):
    """

    :param img: dicom图像的像素值信息
    :param low_window: dicom图像像素值的最低值
    :param high_window: dicom图像像素值的最高值
    :param save_path: 新生成的jpg图片的保存路径
    :return:
    """
    lungwin = np.array([low_window * 1., high_window * 1.])  # 将pydicom解析的像素值转换为array
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 将像素值归一化0-1
    newimg = (newimg * 255).astype('uint8')  # 再转换至0-255，且将编码方式由原来的unit16转换为unit8
    # print(newimg.shape)
    return newimg


def generate_gray(img_path, volume_width, volume_height, mode='png'):
    if mode == 'dcm':
        ds = pydicom.dcmread(img_path)
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = np.uint(ds.pixel_array)
        high = np.max(img)  # 找到最大的
        low = np.min(img)  # 找到最小的
        # 调用函数，开始转换
        img = convert_from_dicom_to_jpg(img, low, high)

        img = np.array(Image.fromarray(np.uint8(img)).resize((volume_width, volume_height), Image.ANTIALIAS))
    else:
        img = np.asarray(np.uint8(Image.open(img_path).convert("L").resize((volume_width, volume_height))))

    return img


def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = 1.5 * (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor


def normalization(img_np, normalization, mode='label'):
    img_tensor = torch.from_numpy(img_np).float()
    if not mode == 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
    return img_tensor


def generate_img_volume(img_folder, folder, dataset_path, volume_width, volume_height):
    img_list = []
    for file in img_folder:
        img_path = os.path.join(dataset_path, 'image', folder, file)
        img_np = generate_gray(img_path, volume_width, volume_height, mode='dcm')
        img_np[img_np<90]=0
        # #********************************todo change the data type
        #
        # ret, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)
        #
        # black = np.zeros((img_np.shape[1], img_np.shape[0]), dtype=np.uint8)
        #
        # contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # for cnt in contours:
        #     hull = cv2.convexHull(cnt)
        #     cv2.drawContours(black, [hull], -1, (255, 255, 255), -1)
        # # print(img_np.shape,black.shape)
        # and_area = cv2.bitwise_and(img_np, black)
        # and_area[and_area < 100] = 0
        # # cv2.imshow(img_path, and_area)
        # # cv2.waitKey(0)
        # img_np=and_area

        # a = 1.3
        # O = float(a) * img_np
        #
        # O[O > 255] = 255  # 大于255要截断为255
        # O[O<120]=0
        # O = np.round(O)
        # O = O.astype(np.uint8)
        img_list.append(img_np)
    img_npy = np.array(img_list)
    return img_npy


def generate_mask_volume(img_folder, folder, dataset_path, volume_width, volume_height, hole=True):
    masks = []
    for file in img_folder:
        file_path = os.path.join(dataset_path, 'label', folder, file)
        img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_grey = cv2.resize(img_grey, (volume_width, volume_height))
        if hole:
            mask = FillHole(img_grey)
        else:
            th1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9,
                                        2)  # 换行符号 \
            mask = 255 - th1
        mask[mask != 0] = 1
        # cv2.imshow('mask slice',mask)
        # cv2.waitKey(1)
        masks.append(mask)
    mask_npy = np.array(masks)
    return mask_npy


def find_no_zero_labels_mask(full_segmentation_map, th_percent, box, num_all, num_crop_all):
    full_segmentation_map[full_segmentation_map > 0] = 1
    num_all_non_zero = full_segmentation_map.sum()
    crop_map = full_segmentation_map[box[0]:box[1], box[2]:box[4], box[3]:box[5]]
    num_crop_non_zero = crop_map.sum()
    thes = num_all_non_zero / num_all
    crop_thes = num_crop_non_zero / num_crop_all

    label_percent = crop_thes / thes

    if label_percent >= th_percent:
        return True
    else:
        return False


def load_medical_image(img_np, box):
    return img_np[box[0]:box[1], box[2]:box[3], box[4]:box[5]]


def save_list(name, list):
    with open(name, 'wb') as fp:
        pickle.dump(list, fp)


def noise(img, mean=0, sigma=0.3):
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out  # 这里也会返回噪声，注意返回值


def subvolume(dataset_path, ilist, type, args):
    dataset_path = os.path.join(dataset_path, 'cbct')
    volume_width = args.batchDim[2]
    volume_height = args.batchDim[1]
    volume_deep = args.batchDim[0]
    th_percent = 1  # 30 for training 10 for val
    resize_width = args.resizeWidth
    resize_height = args.resizeHeight
    if type == 'train':
        overlap = volume_width
        overlap_deep = volume_deep // 2
    else:
        overlap = volume_width
        overlap_deep = volume_deep
    if overlap_deep == 0:
        overlap_deep = 1
    npy_list = []
    ilist = sorted(ilist)
    min_num = 0
    max_num = 0
    mask_max = 0.0
    min_flag = 1
    for ifolder in ilist:

        img_list = sorted(os.listdir(os.path.join(dataset_path, 'image', ifolder)))
        lal_list = sorted(os.listdir(os.path.join(dataset_path, 'label', ifolder)))

        img_np = generate_img_volume(img_list, ifolder, dataset_path, resize_width, resize_height)
        full_segmentation_map = generate_mask_volume(lal_list, ifolder, dataset_path, resize_width, resize_height)
        img_np = noise(img_np, args.noise_mean, args.noise_std)
        img_tensor = normalization(img_np, args.normalization, mode='image')
        # print(img_tensor.max())
        # print(img_tensor.min())
        img_np = img_tensor.numpy()
        # print('big image shape',img_tensor.shape)
        seg_tensor = normalization(full_segmentation_map, args.normalization)
        # print(seg_tensor.max())
        full_segmentation_map = seg_tensor.numpy()

        num = 0
        if not os.path.exists(os.path.join(dataset_path, 'generated')):
            os.makedirs(os.path.join(dataset_path, 'generated'))
        if not os.path.exists(os.path.join(dataset_path, 'generated',
                                           str(volume_deep) + '_' + str(volume_height) + '_' + str(
                                                   volume_width) + '_' + type)):
            os.makedirs(os.path.join(dataset_path, 'generated', str(volume_deep) + '_' + str(volume_height) + '_' + str(
                volume_width) + '_' + type))

        tol_deep = img_np.shape[0]
        num_all, num_crop_all = tol_deep * resize_width * resize_height, volume_deep * volume_width * volume_height
        for deep in range(0, tol_deep - volume_deep, overlap_deep):
            for width in range(0, resize_width + 1 - volume_width, overlap):
                for height in range(0, resize_height + 1 - volume_height, overlap):

                    box = [deep, deep + volume_deep, height, height + volume_height, width, width + volume_width]

                    # print('*****************************')

                    img_tensor = load_medical_image(img_np, box)
                    ann_tensor = load_medical_image(full_segmentation_map, box)
                    # print('img_tensor.shape',img_tensor.shape[0])
                    # print("mask max",ann_tensor.max())
                    # print("mask min",ann_tensor.min())

                    img_npy_path = os.path.join(dataset_path, 'generated',
                                                str(volume_deep) + '_' + str(volume_height) + '_' + str(
                                                    volume_width) + '_' + type,
                                                'id_' + str(ifolder) + '_' + "{0:09d}".format(num) + '.npy')
                    ann_npy_path = os.path.join(dataset_path, 'generated',
                                                str(volume_deep) + '_' + str(volume_height) + '_' + str(
                                                    volume_width) + '_' + type,
                                                'id_' + str(ifolder) + '_' + "{0:09d}".format(num) + '_seg.npy')
                    # print('***********************',ann_npy_path)
                    np.save(img_npy_path, img_tensor)
                    np.save(ann_npy_path, ann_tensor)
                    npy_list.append(tuple([img_npy_path, ann_npy_path]))
                    c = ann_tensor.sum()

                    if min_flag and ann_tensor.sum() > 1.0:
                        min_num = num
                        min_flag = 0
                    if min_flag == 0 and ann_tensor.sum() < 1.0:
                        max_num = num
                    if ann_tensor.sum() > mask_max:
                        mask_max = ann_tensor.sum()
                        mask_max_num = num

                    # ann = np.array(ann_tensor)
                    # for i in range(overlap_deep):
                    #     img=ann[i]
                    #     img[img!=0]=255
                    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    #     cv2.imshow('', img)
                    #     cv2.waitKey(0)

                    num = num + 1

        print("id: %s  sample number: %d  min number: %d  max number: %d mask max number: %d" % (
        ifolder, num, min_num, max_num, mask_max_num))
        min_num = 0
        max_num = 0
        mask_max = 0
        min_flag = 1

    return npy_list

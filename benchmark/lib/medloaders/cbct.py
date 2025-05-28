# -*- coding:utf-8 -*-
from .cbct_utils import *
import torch
from torch.utils.data import Dataset
import lib.utils as utils


class CBCTDataset(Dataset):
    def __init__(self, args, mode, split_id, dataset_path='./datasets'):
        self.mode = mode
        self.root = dataset_path
        self.save_txt_name = os.path.join(dataset_path, 'cbct',
                                          mode + '_' + str(args.batchDim[0]) + '_' + str(args.batchDim[1]) + '_' + str(
                                              args.batchDim[2])) + '.txt'
        # todo
        self.affine = np.zeros((4, 4))
        self.dim = args.batchDim
        self.full_volume = [self.dim[0], self.dim[0], 64, 1]
        self.load = args.loadData
        img_npy_list = sorted(os.listdir(os.path.join(dataset_path, 'cbct', 'image')))

        if self.load:
            self.list = utils.load_list(self.save_txt_name)
            return

        if self.mode == 'train':
            npy_list = img_npy_list[:split_id]
            self.list = subvolume(dataset_path, npy_list, self.mode, args)
        elif self.mode == 'val':
            npy_list = img_npy_list[split_id:]
            self.list = subvolume(dataset_path, npy_list, self.mode, args)

        save_list(self.save_txt_name, sorted(self.list))

    def __getitem__(self, item):
        img_npy_path, lbl_npy_path = self.list[item]
        # if self.mode == 'val':
        #     print("lbl np path", lbl_npy_path)
        return torch.FloatTensor(np.load(img_npy_path)), torch.FloatTensor(np.load(lbl_npy_path)), img_npy_path

    def __len__(self):
        return len(self.list)

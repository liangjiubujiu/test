import numpy as np
import torch
import cv2
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
import os
import lib.utils as utils
import torch.nn as nn
import pydicom
import progressbar
import time
import torchvision
import shutil
import seaborn as sns  # 用于话热图的工具包
import matplotlib.pyplot as plt
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import time
import datetime
import random
class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = args.saveFreq
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.start=str(int(34360))
        self.times=0


    def training(self):

        utils.reproducibility(self.args, seed=1777777)
        global_val_loss=10

        starttime = datetime.datetime.now()

        for epoch in range(self.start_epoch, self.args.nEpochs+1):
            # print('**********************',epoch)
            self.train_epoch(epoch)

            if self.do_validation:

                self.validate_epoch(epoch)


            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            if self.args.save is not None and global_val_loss>val_loss:
                path_state_dict=os.path.join(self.args.save,'best.pkl')
                net_state_dict = self.model.state_dict()
                torch.save(net_state_dict, path_state_dict)
                global_val_loss=val_loss
                best_epoch=epoch

            self.writer.write_end_of_epoch(epoch)
            self.writer.reset('train')
            self.writer.reset('val')

        endtime = datetime.datetime.now()
        print('*****************************')
        print('the best val epoch is :',best_epoch)
        seconds = (endtime - starttime).seconds
        print('the total training seconds are :',seconds)


    def testing(self):

        print('**********************testing************')
        starttime = datetime.datetime.now()
        random.seed(0)
        self.model.load_state_dict(torch.load(os.path.join(self.args.save,'best.pkl')))
        # print(self.model)
        for layer, module in self.model.named_modules():
            if layer=='module.upsacle' or layer=='module.conv3d_l4':
                print('*******')
                print(layer)
                print(module)
                module.register_forward_hook(self.hook_func)
        self.model.eval()

        num=0
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():

                if self.args.dataset_name == 'cbct':
                    input_tensor, target, _ = input_tuple
                    input_tensor = torch.unsqueeze(input_tensor, 1)
                else:
                    img_1, img_2, img_3, img_4, target = input_tuple
                    img_1 = torch.unsqueeze(img_1, 1)
                    img_2 = torch.unsqueeze(img_2, 1)
                    img_3 = torch.unsqueeze(img_3, 1)
                    img_4 = torch.unsqueeze(img_4, 1)
                    input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
                if self.args.cuda:
                    input_tensor, target = input_tensor.cuda(), target.cuda()

                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                output_np = output.detach().cpu().numpy()
                x = output_np
                x = (x - np.min(x)) / (np.max(x) - np.min(x))

                x = x.argmax(axis=1).reshape((-1,x.shape[-2],x.shape[-1]))
                target=target.detach().cpu().numpy().reshape((-1,x.shape[-2],x.shape[-1]))
                # print('x shape', x.shape)
                # print('target shape', target.shape)
                for i in range(x.shape[0]):
                    img = x[i]
                    img[img != 0] = 255
                    img[img != 255] = 0
                    img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    # print(gt_idx,j,i)
                    ground_truth = target[i]
                    ground_truth[ground_truth != 0] = 255
                    ground_truth[ground_truth != 255] = 0
                    ground_truth_rgb = cv2.cvtColor(ground_truth.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    cv2.namedWindow('gt', cv2.WINDOW_NORMAL)
                    cv2.imshow('gt', ground_truth_rgb)
                    cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
                    cv2.imshow("pred", img_rgb)
                    cv2.waitKey(1)  # todo waitkey(0)->waitkey(1)
                    if not os.path.exists(os.path.join(self.args.save,'result')):
                        os.mkdir(os.path.join(self.args.save,'result'))
                    save_path=os.path.join(self.args.save,'result',str(num)+'.jpg')
                    cv2.imwrite(save_path, img_rgb)
                    num = num + 1

        print('***************complete save result************')
        endtime = datetime.datetime.now()
        seconds = (endtime - starttime).seconds
        print('the total inference seconds are :',seconds)

        # print('the generated testing result    ')
        # a=output_total[0]
        # for i in range(len(output_total)-1):
        #     a=np.concatenate((a,output_total[i]),axis=0)
        # res=a
        #
        # if not os.path.exists(os.path.join(self.args.save,"result")):
        #     os.mkdir(os.path.join(self.args.save,"result"))
        # np.save(os.path.join(self.args.save,'result',
        #                      "best_output.npy"), res)
        # print('please cut the result in path ',os.path.join(self.args.save,'result',
        #                      "best_output.npy"))
        # print('the predicted result has shape:',res.shape)
        #
        #

        # print("**********************visualize start******************")
        # gt_np(result_root)

        return



    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):


            if self.args.dataset_name=='cbct':
                input_tensor, target ,_= input_tuple
                input_tensor=torch.unsqueeze(input_tensor, 1)
            else:
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            if self.args.cuda:
                input_tensor, target = input_tensor.cuda(), target.cuda()#[1,2,16,512,512]



            input_tensor.requires_grad = True

            if self.args.model=='CBCTpix':
                self.model.set_input(input_tensor,target)
                loss_dice, per_ch_score = self.model.optimize_parameters()
            else:
                self.optimizer.zero_grad()
                output = self.model(input_tensor)
                loss_dice, per_ch_score = self.criterion(output, target)
                loss_dice.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                print('learning rate',self.optimizer.state_dict()['param_groups'][0]['lr'])


            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)



    def hook_func(self,module, input, output):
        """
        Hook function of register_forward_hook

        Parameters:
        -----------
        module: module of neural network
        input: input of module
        output: output of module
        """


        # if not os.path.exists(os.path.join('/media/mulns/Backup Plus/202104CBCT/feature',str(module).split('(')[0])):
        #     os.mkdir(os.path.join('/media/mulns/Backup Plus/202104CBCT/feature',str(module).split('(')[0]))
        # print('********module',module)
        input_tensor = output.detach().cpu().numpy()

        # print('input_tensor',input_tensor.shape)
        input_tensor=(input_tensor-input_tensor.min())/(input_tensor.max()-input_tensor.min())
        # print(input_tensor.max())
        self.times += 1
        # print(self.times)
        for i in range(input_tensor.shape[0]):
        # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
            for j in range(input_tensor.shape[2]):
                image_tensor = input_tensor[i][0][j]*(255)
                # RGB转BRG
                # image_tensor= cv2.cvtColor(image_tensor, cv2.COLOR_GRAY2BGR)
                # input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
                # self.start = str(34360+self.times*((i+1)*input_tensor.shape[1]+j+1))

                self.start=str(random.randint(1,1000000))
                # print('start',self.start
                if not os.path.exists(os.path.join(self.args.save,'feature')):
                    os.mkdir(os.path.join(self.args.save,'feature'))
                    os.mkdir(os.path.join(self.args.save,'feature','Conv3d'))
                    os.mkdir(os.path.join(self.args.save,'feature','Upsample'))
                filename=os.path.join(self.args.save,'feature',str(module).split('(')[0],self.start+'.jpg')
                # print(filename)
                gray_three_channel = cv2.cvtColor(image_tensor, cv2.COLOR_GRAY2BGR)
                heatmap=cv2.applyColorMap(np.uint8(gray_three_channel),cv2.COLORMAP_RAINBOW)
                # cv2.imshow(filename,heatmap)
                # cv2.waitKey(0)
                cv2.imwrite(filename,heatmap)

        return


    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():

                if self.args.dataset_name == 'cbct':
                    input_tensor, target, _ = input_tuple
                    input_tensor = torch.unsqueeze(input_tensor, 1)
                else:
                    img_1, img_2, img_3, img_4, target = input_tuple
                    img_1 = torch.unsqueeze(img_1, 1)
                    img_2 = torch.unsqueeze(img_2, 1)
                    img_3 = torch.unsqueeze(img_3, 1)
                    img_4 = torch.unsqueeze(img_4, 1)
                    input_tensor = torch.cat((img_1, img_2, img_3, img_4), dim=1)
                if self.args.cuda:
                    input_tensor, target = input_tensor.cuda(), target.cuda()  # [1,2,16,512,512]


                input_tensor.requires_grad = False

                if self.args.model == 'CBCTpix':
                    self.model.set_input(input_tensor, target)
                    output=self.model.forward()
                else:

                    output = self.model(input_tensor)

                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

def gt_np(pred_path):
    gt_root='/media/mulns/data/VV/datasets/MedicalZooPytorch/cbct/generated/16_256_256_val'

    # path_list = sorted(os.listdir(gt_root))
    # seg_list=[]
    # for path in path_list:
    #     if 'seg' in path:
    #
    #         print(os.path.join(gt_root, path))
    #         gt = np.load(os.path.join(gt_root, path))
    #         # print(gt.max())
    #         seg_list.append(gt)
    # gt_numpy=np.array(seg_list)
    # gt=gt_numpy.reshape((-1,gt_numpy.shape[-1],gt_numpy.shape[-2]))




    output_np = np.load(pred_path)
    num=0
    path_list = sorted(os.listdir(gt_root))
    for j in range(output_np.shape[0]):
        path=path_list[j]
        if 'seg' in path:
            print(os.path.join(gt_root, path))
            gt = np.load(os.path.join(gt_root, path))
            x = output_np[j]
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            x = np.transpose(x, axes=(1, 0, 2, 3))
            x = x.argmax(axis=1)
            for i in range(x.shape[0]):

                img = x[i]
                img[img != 0] = 255
                img[img!=255]=0
                img_rgb= cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)



                # print(gt_idx,j,i)
                ground_truth=gt[i]
                ground_truth[ground_truth != 0] = 255
                ground_truth[ground_truth!=255]=0
                ground_truth_rgb = cv2.cvtColor(ground_truth.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                cv2.namedWindow('gt', cv2.WINDOW_NORMAL)
                cv2.imshow('gt', ground_truth_rgb)
                cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
                cv2.imshow("pred", img_rgb)
                cv2.waitKey(0) # todo waitkey(0)->waitkey(1)
                cv2.imwrite(os.path.join(os.path.split(pred_path)[0],str(num)+'.jpg'),img_rgb)
                num=num+1






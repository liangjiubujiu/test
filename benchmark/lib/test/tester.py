import numpy as np
import torch

from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from lib.visual3D_temp.viz_cbct import *

class Tester:
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

        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.args=args

    def testing(self):

        result=self.validate_epoch(100000)
        return result


    def validate_epoch(self, epoch):
        self.model.eval()
        predictions = []
        segment_map=[]

        # input_sub_volumes, segment_map = create_3d_subvol()
        # print(input_sub_volumes.shape, segment_map.shape)
        #
        # sub_volumes = input_sub_volumes.shape[0]


        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                if self.args.dataset_name == 'cbct':
                    input_tensor = torch.unsqueeze(input_tensor, 1)
                if self.args.cuda:
                    input_tensor, target = input_tensor.cuda(), target.cuda()

                input_tensor.requires_grad = False

                output = self.model(input_tensor)

                # print(output_array.shape)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

                predictions.append(output)
                segment_map.append(target)

        predictions = torch.stack(predictions)
        segment_map=torch.stack(segment_map)
        # project back to full volume
        full_vol_predictions = predictions.view(2, self.args.dim[0], self.args.dim[1], self.args.dim[2])
        full_segment_map=segment_map.view(1,self.args.dim[0], self.args.dim[1], self.args.dim[2])
        # full_vol_predictions=predictions
        print("Inference complete", full_vol_predictions.shape)

        # arg max to get the labels in full 3d volume
        _, indices = full_vol_predictions.max(dim=0)
        full_vol_predictions = indices

        print("Class indexed prediction shape", full_vol_predictions.shape, "GT", target.shape)

        # TODO TEST...................
        save_path_2d_fig = self.args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'
        create_2d_views(full_vol_predictions.numpy(), full_segment_map.numpy(), save_path_2d_fig)

        # save_path = self.args.save + '/Pred_volume_epoch_' + str(epoch)
        # save_3d_vol(full_vol_predictions.numpy(), affine, save_path)

        return full_vol_predictions
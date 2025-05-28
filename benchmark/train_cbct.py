# Python libraries
import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
import lib.utils as utils
from lib.losses3D import DiceLoss,GeneralizedDiceLoss
from lib.losses3D import create_loss
import torch


def main():
    args = get_arguments()

    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path=args.dataRoot)

    criterion = DiceLoss(classes=args.classes)
    model, optimizer = medzoo.create_model(args)
    #100=3 epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(5000,15000,500)), gamma=0.9)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    model.cuda()
    # todo weijing delete 30
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=scheduler)

    print("START TRAINING...")

    trainer.training()


    print("START ANALYSE...")
    trainer.testing()



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadData', default=True)  # todo

    parser.add_argument('--batchSz', type=int, default=10)

    parser.add_argument('--model', type=str, default='SWIN',
                        choices=(
                        'SWIN','CBCTpix','SEUNETMINI', 'TransMINI', 'UNET3DMINI', 'UNET3D', 'NONLOCALUNET3D', 'VNET', 'VNET2', 'UNET3D',
                        'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', 'TransBTS', 'ToothPix'))
    parser.add_argument('--batchDim',  type=int, default=(16,256,256))#16,384,384
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--dataset_name', type=str, default="cbct")
    parser.add_argument('--dataRoot', type=str, default='/home/mulns/My_project/VV/ToothZoo/datasets')

    parser.add_argument('--resizeWidth', type=int, default=256)
    parser.add_argument('--resizeHeight', type=int, default=256)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--classes', type=int, default=2)#background
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--terminal_show_freq', default=1)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.95, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--noise_mean', default=0)
    parser.add_argument('--noise_std', default=0.01)
    parser.add_argument('--saveFreq', type=int,default=50)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model_depth', type=int, default=50,help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut',default='B',type=str,help='Shortcut type of resnet (A | B)')
    parser.add_argument('--opt', type=str, default='adam',choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str, default='./run/')

    args = parser.parse_args()
    args.save = './saved_models/' + args.model
    return args

# tensorboard --logdir="/media/mulns/Backup Plus/202104CBCT/unet3dmini/checkpoint"

if __name__ == '__main__':
    main()

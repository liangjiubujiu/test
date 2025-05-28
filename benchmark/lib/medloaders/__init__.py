from torch.utils.data import DataLoader

from .cbct import CBCTDataset


def generate_datasets(args, path):
    params = {'batch_size': args.batchSz,
              'num_workers': 1}
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "cbct":
        total_data = 20
        split_idx = int(split_percent * total_data)
        #
        train_loader = CBCTDataset(args, 'train', dataset_path=path, split_id=split_idx)

        val_loader = CBCTDataset(args, 'val', dataset_path=path, split_id=split_idx)



    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, shuffle=False,**params)#drop_last=True,

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine

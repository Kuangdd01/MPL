import os.path
from functools import partial

from torch.utils.data import DataLoader

from ._glove_tokenizer import build_glove_vocab
from ._glove_tokenizer import build_bert_vocab

def build_dataset(args, tokenizer):
    from ._volta_refer_image_features_reader import RefImageFeaturesH5Reader
    from .refer_dataset import ReferExpressionDataset
    path = os.path.join(args.features_path, args.dataset_name)

    if args.backbone == "clip":
        print("using clip backbone!")
        reader = RefImageFeaturesH5Reader(args, path + "_clip", args.boxfile)
    else:
        reader = RefImageFeaturesH5Reader(args, path, args.boxfile)

    def dataset(split):
        return ReferExpressionDataset(
            args.referoot,
            args.dataset_name,
            split,
            reader,
            tokenizer,
        )

    if args.debug:
        train_dset = dataset(split='val')
        val_dset = train_dset
        test_dset = dataset(split='test')
        testA_dset = dataset(split='testA')
        testB_dset = dataset(split='testB')
    else:
        train_dset = dataset(split='train')
        val_dset = dataset(split='val')
        test_dset = dataset(split='test')
        testA_dset = dataset(split='testA')
        testB_dset = dataset(split='testB')

    return train_dset, val_dset, [test_dset, testA_dset, testB_dset]


def build_mat_dataset(args, tokenizer):
    from ._mat_image_features_reader import MatImageFeaturesH5Reader
    from .mat_entity import MatFlickrGroundingDataset

    def dataset(split):
        mat_root = args.mat_root
        det_name = lambda x: os.path.join(mat_root, f"{x}_detection_dict.json")
        idx_name = lambda x: os.path.join(mat_root, f"{x}_imgid2idx.pkl")
        h5_name = lambda x: os.path.join(mat_root, f"{x}_features_compress.hdf5")

        image_features_reader = MatImageFeaturesH5Reader(args, h5_name(split), idx_name(split), det_name(split))
        return MatFlickrGroundingDataset(dataroot=args.dataroot,
                                         split=split,
                                         image_features_reader=image_features_reader,
                                         tokenizer=tokenizer,
                                         )

    if args.debug:
        train_dset = dataset(split='val')
        val_dset = train_dset
        test_dset = dataset(split='test')
    else:
        train_dset = dataset(split='train')
        val_dset = dataset(split='val')
        test_dset = dataset(split='test')

    return train_dset, val_dset, test_dset


def build_dataloader(args, tokenizer):
    if args.task == 'refer':
        func = build_dataset
    elif args.task == 'phrase':
        func = build_mat_dataset
    else:
        assert False

    train_dset, val_dset, test_dset = func(args, tokenizer)
    if args.num_workers:
        dataloader = partial(DataLoader, batch_size=args.batch_size, pin_memory=True,
                             num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    else:
        dataloader = partial(DataLoader, batch_size=args.batch_size, pin_memory=True,
                             num_workers=args.num_workers)

    train_dl = dataloader(train_dset, collate_fn=train_dset.collate_fn, shuffle=True, drop_last=True)
    val_dl = dataloader(val_dset, collate_fn=val_dset.collate_fn)
    if isinstance(test_dset, list):
        test_dl = []
        for test_dt in test_dset:
            test_dl.append(
                dataloader(test_dt, collate_fn=test_dt.collate_fn)
            )
    else:
        test_dl = dataloader(test_dset, collate_fn=test_dset.collate_fn)
    return train_dl, val_dl, test_dl

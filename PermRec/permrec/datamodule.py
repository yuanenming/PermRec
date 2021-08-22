'''
Description: 
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-08-16 02:06:30
LastEditTime: 2021-08-21 22:25:48
'''

import pytorch_lightning as pl
from .datasets import dataset_factory
from .dataloaders import XLNetDataloader


class PermRecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_code: str = None,
        min_rating: int = 0,
        min_uc: int = None,
        min_sc: int = None,
        seg_len: int = None,
        num_train_seg: int = None,
        num_test_seg: int = None,
        pred_prob: float = None,
        num_workers: int = None,
        test_negative_sampler_code: str = None,
        test_negative_sample_size: int = None,
        train_batch_size: int = None,
        val_batch_size: int = None,
        test_batch_size: int = None

    ):
        super().__init__()
        self.dataset_code = dataset_code
        self.min_rating = min_rating
        self.min_uc = min_uc
        self.min_sc = min_sc
        self.seg_len = seg_len
        self.num_train_seg = num_train_seg
        self.num_test_seg = num_test_seg
        self.pred_prob = pred_prob
        self.num_workers = num_workers
        self.test_negative_sampler_code = test_negative_sampler_code
        self.test_negative_sample_size = test_negative_sample_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        dataset_factory(
            self.dataset_code,
            self.min_rating,
            self.min_uc,
            self.min_sc
            )
        
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.dataset = dataset_factory(
            self.dataset_code,
            self.min_rating,
            self.min_uc,
            self.min_sc
            )
        
        self.dataloader = XLNetDataloader(
            self.dataset,
            self.seg_len,
            self.num_train_seg,
            self.num_test_seg,
            self.pred_prob,
            self.num_workers,
            self.test_negative_sampler_code,
            self.test_negative_sample_size,
            self.train_batch_size,
            self.val_batch_size,
            self.test_batch_size
        )

    def train_dataloader(self):
        return self.dataloader.get_train_loader()
    def val_dataloader(self):
        return self.dataloader.get_valid_loader()
    def test_dataloader(self):
        return self.dataloader.get_test_loader()

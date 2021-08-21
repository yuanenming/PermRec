'''
Description: 
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-07-30 20:10:54
LastEditTime: 2021-08-16 03:06:14
'''

from .negative_samplers import negative_sampler_factory

from abc import *


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self,
            dataset,
            test_negative_sampler_code,
            test_negative_sample_size
            ):
        save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        test_negative_sampler = negative_sampler_factory(test_negative_sampler_code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         test_negative_sample_size,
                                                         save_folder)

        self.test_negative_samples = test_negative_sampler.get_negative_samples()


    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_valid_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass
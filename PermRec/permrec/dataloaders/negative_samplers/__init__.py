'''
Description: 
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-08-16 03:08:17
LastEditTime: 2021-08-16 03:08:17
'''
from .popular import PopularNegativeSampler
from .random import RandomNegativeSampler


NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, train, val, test, user_count, item_count, sample_size, save_folder):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, test, user_count, item_count, sample_size, save_folder)

'''
Description: XLNet Dataloader
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-07-30 20:25:14
LastEditTime: 2021-08-21 20:00:25
'''

from .base import AbstractDataloader

import torch
import numpy as np
import torch.utils.data as data_utils


class XLNetDataloader(AbstractDataloader):
    def __init__(
            self,
            dataset,
            seg_len,
            num_seg,
            pred_prob,
            num_workers,
            test_negative_sampler_code,
            test_negative_sample_size,
            train_batch_size,
            val_batch_size,
            test_batch_size
        ):
        super().__init__(dataset,
            test_negative_sampler_code,
            test_negative_sample_size)
        self.seg_len = seg_len
        self.num_seg = num_seg
        self.pred_prob = pred_prob
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=self.num_workers)
        return dataloader

    def _get_train_dataset(self):
        dataset = XLNetTrainDataset(self.train, self.seg_len, self.num_seg, self.pred_prob)
        return dataset

    def get_valid_loader(self):
        return self._get_eval_loader(mode='val')

    def get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.val_batch_size if mode == 'val' else self.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=self.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = XLNetEvalDataset(self.train, answers, self.seg_len, self.num_seg, self.test_negative_samples)
        return dataset


class XLNetTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, seg_len, num_seg, pred_prob):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.seg_len = seg_len
        self.num_seg = num_seg
        self.max_len = seg_len * num_seg
        self.pred_prob = pred_prob
        self.max_pred = int(seg_len * pred_prob)

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        if len(seq) > self.max_len:
            begin_idx = np.random.randint(0, len(seq)-self.max_len+1)
            seq = seq[begin_idx:begin_idx+self.max_len]
        
        ret = {
            "user_id" : [torch.LongTensor([index])],
            "input_ids": [],
            "labels" : [],
            "perm_mask": [],
            "target_mapping": []
        }

        for i in range(self.num_seg):
            if i*self.seg_len >= len(seq):
                ret["input_ids"].insert(0, torch.zeros(self.seg_len).long())
                ret["labels"].insert(0, torch.zeros(self.max_pred).long())
                ret['perm_mask'].insert(0, torch.ones(self.seg_len, self.seg_len).long())
                ret["target_mapping"].insert(0, torch.zeros(self.max_pred, self.seg_len))            
            else:
                seg = seq[max(0, len(seq) - (i+1)*self.seg_len): len(seq) - i*self.seg_len]
                goal_num_predict = max(1, int(self.pred_prob * len(seg)))

                # sample prediction masks
                target_mask = self._sample_mask(seg, goal_num_predict=goal_num_predict)
                # padding
                padding_len = self.seg_len - len(seg)
                seg = [0] * padding_len + seg
                target_mask = torch.Tensor([False] * padding_len + list(target_mask)).bool()

                # make permutation masks
                perm_mask = self._permute(seg, target_mask)

                # target mapping
                target_mapping = torch.eye(len(seg), dtype=torch.float32)[target_mask]
                padding_len = self.max_pred - target_mapping.shape[0]
                paddings = torch.zeros([padding_len, self.seg_len], dtype=target_mapping.dtype)
                target_mapping = torch.cat([paddings, target_mapping], dim=0)

                # labels
                labels = torch.LongTensor(seg)[target_mask]
                paddings = torch.zeros(padding_len, dtype=labels.dtype)
                labels = torch.cat([paddings, labels], dim=0)

                ret["input_ids"].insert(0, torch.LongTensor(seg))
                ret["labels"].insert(0, labels)
                ret['perm_mask'].insert(0, perm_mask)
                ret["target_mapping"].insert(0, target_mapping)
                
        return ret        

    def _getseq(self, user):
        return self.u2seq[user]

    def _sample_mask(self, input, goal_num_predict=None):
        input_len = len(input)
        index = np.random.choice(input_len, goal_num_predict, replace=False)
        mask = np.array([False] * input_len, dtype=np.bool)
        mask[index] = True
        return mask

    def _permute(self, input, is_masked):
        """
        Sample a permutation of the factorization order, and create an
        attention mask accordingly.
        Args:
        inputs: int64 Tensor in shape [seq_len], input ids.
        targets: int64 Tensor in shape [seq_len], target ids.
        is_masked: bool Tensor in shape [seq_len]. True means being selected
        for partial prediction.
        perm_size: the length of longest permutation. Could be set to be reuse_len.
        Should not be larger than reuse_len or there will be data leaks.
        seq_len: int, sequence length.
        """
        # Generate permutation indices
        seq_len = len(input)
        index = torch.arange(seq_len, dtype=torch.int64)
        index = index[torch.randperm(index.shape[0])]

        # Set the permutation indices of non-masked (& non-funcional) tokens to the
        # smallest index (-1):
        # (1) they can be seen by all other positions
        # (2) they cannot see masked positions, so there won"t be information leak
        smallest_index = -torch.ones([seq_len], dtype=torch.int64)

        # put -1 if `non_mask_tokens(real token not cls or sep)` not permutation index
        rev_index = torch.where(~is_masked, smallest_index, index)

        # Create `perm_mask`
        # `target_tokens` cannot see themselves
        self_rev_index = torch.where(is_masked, rev_index, rev_index + 1)

        # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
        # 0: can attend if i > j or j is non-masked
        perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) &  is_masked
        perm_mask = perm_mask.type(torch.float32)

        return perm_mask

    
class XLNetEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, seg_len, num_seg, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.seg_len = seg_len
        self.num_seg = num_seg
        self.max_len = seg_len * num_seg
        self.negative_samples = negative_samples

    def __len__(self):
        # to fast evaluate, set a max evaluation size
        # return min(len(self.users), 50000)
        return len(self.users)


    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        # Add dummy token at the end (no attention on this one)
        seq = seq + [1]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        ret = {
            "user_id" : [torch.LongTensor([index])],
            "input_ids": [],
            "labels" : [torch.LongTensor(labels)],
            "perm_mask": [],
            "target_mapping": [],
            "candidates": [torch.LongTensor(candidates)],
        }

        for i in range(self.num_seg):
            seg = seq[i*self.seg_len:(i+1)*self.seg_len]

            # Build permutation mask so that previous tokens don't see last token
            perm_mask = torch.zeros((self.seg_len, self.seg_len), dtype=torch.float)

            # We'll only predict the last token
            target_mapping = torch.zeros((1, self.seg_len), dtype=torch.float)
            if i == self.num_seg-1:
                perm_mask[:, -1] = 1.0
                target_mapping[0, -1] = 1.0
            ret['input_ids'].append(torch.LongTensor(seg))
            ret['perm_mask'].append(perm_mask)
            ret['target_mapping'].append(target_mapping)

        return ret
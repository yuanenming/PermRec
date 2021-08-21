'''
Description: 
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-08-15 16:24:01
LastEditTime: 2021-08-21 19:59:37
'''
from typing import List
import torch
import pytorch_lightning as pl
from .models import XLNetModel, DotProductPredictionHead
from .utils import LabelSmoothSoftmaxCEV1, recalls_and_ndcgs_for_ks


import os


class PermRecModel(pl.LightningModule):

    def __init__(self,
            xlnet: XLNetModel,
            label_smooth: int = 0.0,

        ):
        super().__init__()
        self.xlnet = xlnet
        self.head = DotProductPredictionHead(xlnet.d_model, xlnet.num_items, self.xlnet.item_embedding)
        self.loss = LabelSmoothSoftmaxCEV1(lb_smooth=label_smooth, ignore_index=0)

    def forward(self, user_id, input_ids, perm_mask, target_mapping, use_mems=True):
        mems = None
        mems_mask = None
        outputs = []
        assert isinstance(input_ids, List)
        for i in range(len(input_ids)):
            input_mask = (input_ids[i] == 0).float()
            output, mems = self.xlnet(
                user_id = user_id,
                input_ids = input_ids[i],
                mems=mems,
                perm_mask = perm_mask[i],
                target_mapping = target_mapping[i],
                input_mask = input_mask,
                use_mems=use_mems,
                mems_mask=mems_mask
            )
            mems_mask = input_mask
            # mems and mems_mask are used for next segment
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)   

        return outputs
        
    def training_step(self, batch, batch_idx):
        user_id = batch['user_id'][0]
        input_ids = batch['input_ids']
        perm_mask = batch['perm_mask']
        target_mapping = batch['target_mapping']

        outputs = self(user_id, input_ids, perm_mask, target_mapping)
        outputs = outputs.view(-1, outputs.size(-1))  # BT x H

        labels = torch.cat(batch['labels'], dim=1)
        labels = labels.view(-1)  # BT
        logits = self.head(outputs) # BT x H
        loss = self.loss(logits, labels)
        loss = loss.unsqueeze(0)

        return {'loss':loss}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.cat([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        user_id = batch['user_id'][0]
        input_ids = batch['input_ids']
        perm_mask = batch['perm_mask']
        target_mapping = batch['target_mapping']

        outputs = self(user_id, input_ids, perm_mask, target_mapping)

        # get scores (B x C) for evaluation
        last_outputs = outputs[:, -1, :]
        candidates = batch['candidates'][0] # B x C
        logits = self.head(last_outputs, candidates) # B x C

        labels = torch.cat(batch['labels'], dim=1)
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])

        return metrics
    
    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in validation_step_outputs:
                tmp.append(o[k])
            self.log(f'Val:{k}', torch.Tensor(tmp).mean())

    # def test_step(...):
    #     pass

    # def test_step_end(...):
    #     pass

    # def test_epoch_end(...):
    #     pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

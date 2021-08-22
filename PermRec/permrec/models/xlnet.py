'''
Description: XLNet model: memory, PLM, relative position enabled, modified from huggingface XLNetModel
Author: Enming Yuan
email: yem19@mails.tsinghua.edu.cn
Date: 2021-08-15 20:59:54
LastEditTime: 2021-08-22 11:24:44
'''

# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 PyTorch XLNet model.
"""
import torch
from torch import nn
import pytorch_lightning as pl


"""
TODO:
"""


class XLNetRelativeAttention(pl.LightningModule):
    def __init__(self,
            d_model: int = None,
            n_head: int = None,
            layer_norm_eps: float = 1e-8,
            dropout: float = None,
            ):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                f"The hidden size ({d_model}) is not a multiple of the number of attention "
                f"heads ({n_head}"
            )

        d_head = int(d_model / n_head)
        self.scale = 1 / (d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(d_model, n_head, d_head))
        self.k = nn.Parameter(torch.FloatTensor(d_model, n_head, d_head))
        self.v = nn.Parameter(torch.FloatTensor(d_model, n_head, d_head))
        self.r = nn.Parameter(torch.FloatTensor(d_model, n_head, d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(n_head, d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(n_head, d_head))

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        attn_mask=None,
    ):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # merge attention scores and perform masking
        attn_score = (ac + bd) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.float16:
                attn_score = attn_score - 65500 * torch.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * torch.einsum("ijbn->bnij", attn_mask)

        # attention probability
        attn_prob = nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing. (back to `d_model`)"""
        attn_out = attn_vec.reshape(attn_vec.shape[0], attn_vec.shape[1], -1)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        mems=None,
        target_mapping=None,
    ):
        if g is not None:
            # Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content-based key head
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)

            # content-based value head
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)

            # position-based key head
            k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r)

            # h-stream
            # content-stream query head
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                attn_mask=attn_mask_h,
            )

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            # g-stream
            # query-stream query head
            q_head_g = torch.einsum("ibh,hnd->ibnd", g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = torch.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    attn_mask=attn_mask_g,
                )

                attn_vec_g = torch.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    attn_mask=attn_mask_g,
                )

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

        else: 
            # Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = torch.cat([mems, h], dim=0)
            else:
                cat = h

            # content heads
            q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
            k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
            v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)

            # positional heads
            # type casting for fp16 support
            k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                attn_mask=attn_mask_h,
            )

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        return outputs


class XLNetFeedForward(nn.Module):
    def __init__(self,
            d_model: int = None,
            d_inner: int = None,
            layer_norm_eps: float = 1e-8,
            dropout: float = None,
            activation_type: str = None
            ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_1 = nn.Linear(d_model, d_inner)
        self.layer_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation_type == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_type == 'gelu':
            self.activation_function = nn.GELU()

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):
    def __init__(self,
            d_model: int = None,
            n_head: int = None,
            d_inner: int = None,
            layer_norm_eps: float = 1e-8,
            dropout: float = None,
            activation_type: str = None
            ):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(
            d_model=d_model,
            n_head=n_head,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout
        )
        self.ff = XLNetFeedForward(
            d_model=d_model,
            d_inner=d_inner,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            activation_type=activation_type
        )

    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        mems=None,
        target_mapping=None,
    ):
        output_h, output_g  = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            mems=mems,
            target_mapping=target_mapping,
        )

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        outputs = output_h, output_g
        return outputs

class NeuralGatingNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_s = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_u = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,  h_s, h_user):
        logits_s = torch.einsum("ibd,dh->ibh", h_s , self.w_s)
        logits_u = torch.einsum("ibd,dh->ibh", h_user, self.w_u)
        G = self.sigmoid(logits_s + logits_u)
        h_s = torch.einsum("ibd,ibd->ibd", h_s, G)
        h_u = torch.einsum("ibd,ibd->ibd", h_user, 1-G)
        return h_s + h_u

class XLNetModel(nn.Module):
    def __init__(self,
            d_model: int = None,
            n_head: int = None,
            d_inner: int = None,
            layer_norm_eps: float = 1e-8,
            dropout: float = None,
            activation_type: str = None,
            clamp_len: int = None,
            n_layer: int = None,
            num_items: int = None,
            device: str = None,
            initializer_range: float = 0.02
        ):
        super().__init__()

        self.initializer_range = initializer_range

        self.d_model = d_model
        self.num_items = num_items
        self.activation_type = activation_type

        self.clamp_len = clamp_len

        self.item_embedding = nn.Embedding(num_items+1, d_model, padding_idx=0)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1, 1, d_model))
        self.layer = nn.ModuleList(
            [
                XLNetLayer(
                    d_model=d_model,
                    n_head=n_head,
                    d_inner=d_inner,
                    layer_norm_eps=layer_norm_eps,
                    dropout=dropout,
                    activation_type=activation_type) for _ in range(n_layer)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.device = device
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, XLNetRelativeAttention):
            for param in [
                module.q,
                module.k,
                module.v,
                module.r,
                module.r_r_bias,
                module.r_w_bias,
            ]:
                param.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, XLNetModel):
            module.mask_emb.data.normal_(mean=0.0, std=self.initializer_range)
    
    def cache_mem(self, curr_out):
        # cache hidden states into memory.
        new_mem = curr_out
        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        beg, end = klen, -qlen

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        if self.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(self.device)
        return pos_emb

    def forward(
        self,
        input_ids=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        input_mask=None,
        use_mems=None,
        mems_mask=None
    ):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        input_ids = input_ids.transpose(0, 1).contiguous()
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        mems_mask = mems_mask.transpose(0, 1).contiguous() if mems_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        # dealing with mask: input_mask & perm_mask & mems_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + input_mask[:,None] + perm_mask
            if mlen > 0:
                if mems_mask is not None:
                    mems_mask = mems_mask.expand(data_mask.shape[0], mlen, bsz).to(data_mask)
                else: # all mems can be attended to
                    mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            attn_mask = data_mask[:, :, :, None]
            attn_mask = (attn_mask > 0).float()
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            raise RuntimeError

        # Word embeddings and prepare h & g hidden states
        item_emb_k = self.item_embedding(input_ids)
        output_h = self.dropout(item_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # calculation
        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        for i, layer_module in enumerate(self.layer):
            if use_mems:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h),)

            output_h, output_g = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                mems=mems[i],
                target_mapping=target_mapping
            )

        output = self.dropout(output_g if output_g is not None else output_h)

        if not use_mems:
            new_mems = None

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()

        return output, new_mems

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
# Modified from VisionZip (https://github.com/dvlab-research/VisionZip)
# Copyright 2024 Jinhong Deng
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder

from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward

sys.path.append('/LLaVA/llava/model/methods_utils')
import submodular_function, submodular_optimizer



class CLIPVisionTower_OPTP(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        # import time
        # start_time = time.time()
        # import ipdb; ipdb.set_trace()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2]
            metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            dominant_num =  self.vision_tower._info["dominant"]

            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]
            cls_attention_sum = cls_attention.sum(dim=1)

            image_features = hidden_states[:, cls_idx + 1:]
            bs = image_features.shape[0]
            dominant_num = int(dominant_num /bs)
            selected_idx, _ = OPTP(image_features, dominant_num, cls_attention_sum)
            selected_idx += 1

            all_indices = selected_idx 
            mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
            dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
            
            hidden_states_save = dominant_tokens

        return hidden_states_save, all_indices

def OPTP(visual_feature_vectors, num_selected_token, cls_attn=None):
    """
     基于正交投影（Orthogonal Projection）的 Token 选择机制。
     输入输出与原函数保持一致。
     """
    B, N, D = visual_feature_vectors.shape
    device = visual_feature_vectors.device
    dtype = visual_feature_vectors.dtype
    # 1. 预计算余弦相似度（保持输出对齐）
    # 注意：在正交化逻辑中，我们直接操作特征，不再强依赖原始相似度矩阵进行惩罚
    norm = visual_feature_vectors.norm(dim=-1, keepdim=True)
    norm_vectors = visual_feature_vectors / (norm + 1e-6)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    # 2. 初始得分计算
    alpha = float(os.environ.get('ALPHA', '1.0'))
    # 基础质量得分 q
    base_q = (cls_attn ** alpha) if cls_attn is not None else torch.ones(B, N, dtype=dtype, device=device)
    # 3. 正交投影选择逻辑
    selected_idx = torch.empty(B, num_selected_token, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    # current_x 将在迭代中不断减去已选维度的分量，变为“残差特征”
    current_x = visual_feature_vectors.clone()
    for i in range(num_selected_token):

        residual_norm = torch.norm(current_x, dim=-1)
        # 动态得分 = 初始重要性 + 剩余能量占比
        current_q = base_q + residual_norm
        # 屏蔽掉已经选过的点
        if i > 0:
            for j in range(i):
                min_value = torch.finfo(current_q.dtype).min
                current_q[batch_indices, selected_idx[:, j]] = min_value

        # 选出当前最优 Token
        best_idx = current_q.argmax(dim=1)
        selected_idx[:, i] = best_idx
        # --- 核心正交化步骤 ---
        # 提取选中的特征向量 v: [B, D]
        v = current_x[batch_indices, best_idx].unsqueeze(1)  # [B, 1, D]
        # 计算单位正交基 u = v / |v|
        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True) + 1e-8
        # 计算所有 Token 在 v 方向上的投影分量，并减去它
        # 投影公式: x_new = x - ( (x·v) / (v·v) ) * v
        # 利用 bmm 实现 Batch 级别的点积
        dot_product = torch.bmm(current_x, v.transpose(1, 2))  # [B, N, 1]
        projection = (dot_product / v_norm_sq) * v  # [B, N, D]
        lamana = 0.5
        current_x = current_x - lamana * projection  # 更新残差空间
    return selected_idx, cosine_simi




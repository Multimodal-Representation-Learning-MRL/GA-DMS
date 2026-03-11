from model import objectives
from .clip_model import ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

def attention_layer(q, k, v, num_heads=1, attn_mask=None):
    "Compute 'Scaled Dot Product Attention'"
    tgt_len, bsz, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    # device = attn_mask.device
    # q = q.to(device)
    # k = k.to(device)
    # v = v.to(device)
        
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        # print(attn_mask.shape)
        # print(attn_output_weights.shape)
        # print("attn_mask dtype:", attn_mask)
        # print("attn_output_weights dtype:", attn_output_weights)
        attn_output_weights += attn_mask
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    return attn_output, attn_output_weights

def multi_scale_pooling(features, scales=[1, 2]):
    """
    对特征进行多尺度池化
    Args:
        features: 输入特征 [seq_len, batch_size, dim]
        scales: 池化尺度列表
    Returns:
        融合后的特征
    """
    batch_size = features.shape[1]
    dim = features.shape[2]
    pooled_features = []
    
    for scale in scales:
        # 计算完整的段数
        full_segments = features.shape[0] // scale
        seq_len = full_segments * scale
        
        # 重塑特征以进行池化
        reshaped = features[:seq_len].view(full_segments, scale, batch_size, dim)
        # 对每个尺度进行平均池化
        pooled = reshaped.mean(dim=1)  # [full_segments, batch_size, dim]
        
        # 处理剩余部分
        if seq_len < features.shape[0]:
            remaining = features[seq_len:]
            remaining_pooled = remaining.mean(dim=0, keepdim=True)  # [1, batch_size, dim]
            pooled = torch.cat([pooled, remaining_pooled], dim=0)
        
        # 上采样回原始长度
        upsampled = F.interpolate(
            pooled.permute(1, 2, 0),  # [batch_size, dim, seq_len//scale + 1]
            size=features.shape[0],
            mode='linear',
            align_corners=False
        ).permute(2, 0, 1)  # [seq_len, batch_size, dim]
        pooled_features.append(upsampled)
    
    # 如果没有有效的池化结果，返回原始特征
    if not pooled_features:
        return features
    
    # 融合所有尺度的特征
    return torch.stack(pooled_features).mean(dim=0)
def enhanced_sim_qk(q, k, eos_position, scales=[1, 2]):
    """
    增强的相似度计算，包含多尺度特征 (Handles Batches)
    Args:
        q: 查询向量 [seq_len, batch_size, dim]
        k: 键向量 [seq_len, batch_size, dim]
        eos_position: EOS标记位置 Tensor[batch_size]
        scales: 特征尺度列表
    Returns:
        多尺度融合的相似度分数 [batch_size, seq_len]
    """
    batch_size = q.shape[1]
    seq_len = q.shape[0]
    device = q.device

    # 对q和k进行多尺度池化
    q_multi = multi_scale_pooling(q, scales)
    k_multi = multi_scale_pooling(k, scales)

    # 获取 batch_size 个 EOS token 的 Q 表示
    # q_multi shape: [seq_len, batch_size, dim]
    # eos_position shape: [batch_size]
    # 
    q_cls = F.normalize(q_multi[eos_position, :, :], dim=-1) # Shape [batch_size, dim]
    diagonal_elements = torch.diagonal(q_cls, dim1=0, dim2=1) # Shape [dim, batch_size]
    q_cls_norm = diagonal_elements.permute(1, 0).unsqueeze(1) # 形状变为 torch.Size([2 ,1 ,512])


    # 获取所有 K token 的表示 (K Patch embeddings)
    # k_multi shape: [seq_len, batch_size, dim]
    k_patch_norm = F.normalize(k_multi, dim=-1) # Shape [seq_len, batch_size, dim]
    k_patch_norm = k_patch_norm.permute(1, 0, 2) # Shape [batch_size, seq_len, dim]

    # 计算余弦相似度 (batch dot product)
    # Want result [batch_size, seq_len]
    # q_cls_norm.unsqueeze(1): [batch_size, 1, dim]
    # k_patch_norm.transpose(1, 2): [batch_size, dim, seq_len]
    # cosine_qk = torch.bmm(q_cls_norm.unsqueeze(1), k_patch_norm.transpose(1, 2)).squeeze(1)
    # Simpler: Element-wise product and sum
    cosine_qk = (q_cls_norm * k_patch_norm).sum(-1) # Shape [batch_size, seq_len]

    # 归一化 per batch item
    min_vals = torch.min(cosine_qk, dim=1, keepdim=True)[0]
    max_vals = torch.max(cosine_qk, dim=1, keepdim=True)[0]
    cosine_qk_normalized = (cosine_qk - min_vals) / (max_vals - min_vals + 1e-8)

    return cosine_qk_normalized
def sim_qk(q, k,eos_position): #torch.Size([77, 2, 512])
    q_cls = F.normalize(q[eos_position,:,:], dim=-1) # torch.Size([2, 2, 512]) 
    diagonal_elements = torch.diagonal(q_cls, dim1=0, dim2=1) # torch.Size([512, 2])
    k_patch = F.normalize(k, dim=-1) # torch.Size([77, 2, 512])

    a_prep = diagonal_elements.permute(1, 0).unsqueeze(1) # 形状变为 torch.Size([2 ,1 ,512])

    # 调整 b 的形状以匹配 a_prep
    b_prep = k_patch.permute(1,0,2)  # 形状变为 torch.Size([2, 77, 512])

    # 执行批次矩阵乘法
    c = (a_prep * b_prep)  # 结果形状为 [77, 512, 2]

    # # 转置最后两维，以得到所需的输出形状 [2, 77, 512]
    # c = c.transpose(1, 2)  # 现在形状变为 [77, 2, 512]
    # c = c.transpose(0, 1)  # 最终形状 [2, 77, 512]
    cosine_qk = c.sum(-1)
    cosine_qk = (cosine_qk - cosine_qk.min(dim=1, keepdim=True)[0]) / (cosine_qk.max(dim=1, keepdim=True)[0] - cosine_qk.min(dim=1, keepdim=True)[0])
    # cosine_qk = (q_cls * k_patch).sum(-1)  
    # cosine_qk = (cosine_qk-cosine_qk.min()) / (cosine_qk.max()-cosine_qk.min())
    return cosine_qk
class IRRA2(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def clip_encode_text_dense(self,ori_token,eos,n):
        x = ori_token.half()
        # device = ori_token.device
        #####################
        attns = []
        atten_outs = []
        vs = []
        qs = []
        ks = []
        for TR in self.base_model.transformer.resblocks[-n:]: # .module
            x_in = x
            x = TR.ln_1(x_in)
            linear = torch._C._nn.linear  
            # print("x dtype:", x.dtype)
            # print("TR.attn.in_proj_weight dtype:", TR.attn.in_proj_weight.dtype)
            # print("TR.attn.in_proj_bias dtype:", TR.attn.in_proj_bias.dtype) 
            q, k, v = linear(x.half(), TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
            attn_output, attn = attention_layer(q, k, v, 1, attn_mask=self.base_model.build_attention_mask().cuda()) # .module# num_head=1
            attns.append(attn)
            atten_outs.append(attn_output)
            vs.append(v)
            qs.append(q)
            ks.append(k)

            x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)       
            x = x_after_attn + x_in
            x = x + TR.mlp(TR.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD # torch.Size([2, 77, 512])
        x = self.base_model.ln_final(x).type(self.base_model.dtype) #.module

        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos] @ self.base_model.text_projection
        return x, (qs, ks, vs), attns, atten_outs
    def grad_eclip_enhanced_batched(self,c, qs, ks, vs, attn_outputs, attns, eos_position, scales=[1, 2]):
        """
        Batched Grad-ECLIP with multi-scale features and attention modulation.
        Args:
            c: Similarity matrix [batch_size, batch_size]
            qs, ks, vs: List of tensors [seq_len, batch_size, dim]
            attn_outputs: List of tensors [seq_len, batch_size, dim]
            attns: List of attention maps [batch_size, 1, seq_len, seq_len]
            eos_position: EOS indices [batch_size]
            scales: List of scales for multi-scale pooling
        Returns:
            Importance map [batch_size, num_tokens_effective]
        """
        batch_size = c.shape[0]
        seq_len = qs[0].shape[0]
        device = c.device
        num_layers = len(qs)

        # We care about the self-similarity (diagonal of c)
        # c_diag = torch.diag(c)
        # target_scalar = c_diag.sum() # Aggregate target for simpler grad calculation
        grad_outputs = torch.eye(c.size(-1), device=c.device, dtype=c.dtype)
        tmp_maps = []
        # Use enumerate to get layer_index
        for layer_index, (q, k, v, attn_output) in enumerate(zip(qs, ks, vs, attn_outputs)):
            retain_graph = layer_index < num_layers - 1 # Keep graph until last layer
            # 计算梯度 w.r.t the aggregated diagonal similarity
            grad = torch.autograd.grad(
                c,
                attn_output,
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                allow_unused=True # Allow unused if some parts of graph disconnected
            )[0]  # [seq_len, batch_size, dim]

            if grad is None:
                print(f"Warning: Gradient is None for layer {layer_index}. Skipping layer.")
                # Append zeros map of correct size or handle appropriately
                # For now, let's append zeros map of expected size [batch_size, seq_len-1] (assuming BOS removed)
                # Determine max eos position to define size
                max_eos = torch.max(eos_position)
                tmp_maps.append(torch.zeros((batch_size, max_eos - 1), device=device, dtype=q.dtype))
                continue

            # 对梯度进行多尺度池化
            # grad_multi = multi_scale_pooling(grad, scales)
            grad_cls = grad[eos_position,:,:]
            diagonal_elements = torch.diagonal(grad_cls, dim1=0, dim2=1).T.unsqueeze(1)
            # 计算增强的相似度 (using batched function)
            # Note: enhanced_sim_qk needs eos_position passed to it
            cosine_qk = enhanced_sim_qk(q, k, eos_position, scales) # Shape [batch_size, seq_len]

            # --- Calculate Importance --- 
            # Option: Use only grad at EOS position (like original)
            # grad_cls = grad_multi[eos_position, torch.arange(batch_size, device=device), :] # Shape [batch_size, dim]
            # v_perm = v.permute(1, 0, 2) # Shape [batch_size, seq_len, dim]
            # importance_raw = (grad_cls.unsqueeze(1) * v_perm * cosine_qk.unsqueeze(-1)).sum(-1) # Shape [batch_size, seq_len]

            # Option: Use grad from all positions (more comprehensive)
            # grad_all_pos = grad_multi.permute(1, 0, 2) # Shape [batch_size, seq_len, dim]
            v_perm = v.permute(1, 0, 2) # Shape [batch_size, seq_len, dim]
            # Calculate importance term per token position
            # grad * v * sim_qk -> [batch_size, seq_len, dim] * [batch_size, seq_len, dim] * [batch_size, seq_len, 1]
            importance_term = diagonal_elements * v_perm * cosine_qk.unsqueeze(-1)
            importance_raw = importance_term.sum(-1) # Sum over dim -> Shape [batch_size, seq_len]

            # --- Attention Modulation --- 
            # Get attention weights FROM EOS position TO other tokens
            # attns[layer_index] shape: [batch_size, seq_len, seq_len] (Head dimension was averaged out)
            current_attn_from_eos = attns[layer_index][torch.arange(batch_size), eos_position, :] # Remove head index 0

            # Create masks for valid tokens (excluding BOS and padding beyond EOS)
            token_indices = torch.arange(seq_len, device=device)
            mask = (token_indices > 0) & (token_indices < eos_position.unsqueeze(1))
            mask = mask.to(q.dtype) # Convert mask to float for multiplication

            # Extract scores and attention for relevant tokens using the mask
            importance_masked = importance_raw * mask
            attn_weights_masked = current_attn_from_eos * mask

            # Normalize attention weights over relevant tokens for each batch item
            attn_norm = importance_raw.sum(dim=1, keepdim=True)
            attn_weights_normalized = attn_weights_masked / (attn_norm + 1e-8)

            # Modulate by attention
            importance_modulated = importance_masked * attn_weights_normalized

            # Append the map (we'll handle consistent slicing later)
            tmp_maps.append(importance_modulated)

        # --- Aggregate Layers --- 
        # Stack maps: list of [batch_size, seq_len] -> [num_layers, batch_size, seq_len]
        if not tmp_maps: # Handle case where all gradients were None
            print("Error: No valid importance maps generated.")
            max_eos = torch.max(eos_position)
            return torch.zeros((batch_size, max_eos -1), device=device, dtype=qs[0].dtype) # Return dummy zero map

        stacked_maps = torch.stack(tmp_maps, dim=0)
        # Sum across layers
        emap_aggregated = F.relu_(stacked_maps.sum(0)) # Shape [batch_size, seq_len]

        # --- Final Processing & Normalization --- 
        # Apply mask again to ensure only valid tokens are considered after aggregation
        # Use the mask derived from the original eos_position tensor
        token_indices = torch.arange(seq_len, device=device)
        final_mask = (token_indices > 0) & (token_indices < eos_position.unsqueeze(1))
        emap_final = emap_aggregated * final_mask.to(emap_aggregated.dtype)

        # Normalize per batch item over the valid tokens
        # emap_sum = emap_final.sum(dim=1, keepdim=True)
        # emap_normalized = emap_final / (emap_sum + 1e-8)


        min_val = torch.min(emap_final)
        max_val = torch.max(emap_final)
        # 归一化 grad_emap 到区间 [0, 1]
        normalized_grad_emap = (emap_final - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(emap_final)
        return normalized_grad_emap
    def grad_eclip(self,c, qs, ks, vs, attn_outputs, eos_position):
        ## gradient on last attention output
        tmp_maps = []

        grad_outputs = torch.eye(c.size(-1), device=c.device, dtype=c.dtype)
        eos_position = eos_position.to(c.device)
        for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
            grad = torch.autograd.grad(
                c,
                attn_output,
                grad_outputs=grad_outputs,
                retain_graph=True if len(tmp_maps) < len(attn_outputs) - 1 else False)[0]
            grad = grad.to(c.device)
            grad_cls = grad[eos_position,:,:]
            diagonal_elements = torch.diagonal(grad_cls, dim1=0, dim2=1).T.unsqueeze(1)
            # just use the gradient on the cls token position  
            cosine_qk = sim_qk(q, k,eos_position) # torch.Size([2, 77])
            # print("[cosine_qk]:", cosine_qk.shape) # 77
            v = v.permute(1,0,2)
            tmp_maps.append((diagonal_elements * v[:,:,:] * cosine_qk[:,:,None]).sum(-1))

        emap = (F.relu_(torch.stack(tmp_maps, dim=0).sum(0))) # torch.Size([2, 77])
        # emap = emap[1:eos_position].flatten()
        emap = emap / emap.max(dim=1, keepdim=True).values
        # 参数设置
        # k = torch.tensor(10.0).to(c.device)  # 控制斜率
        # threshold = torch.tensor(0.5).to(c.device)  # 控制中点
        # emap = sigmoid(emap, k, threshold)
        return emap
    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        ori_text = batch['caption_ids_ori']
        noise_text = batch['noise_text']
        # mlm_text = batch['mlm_text']
        eos_position = ori_text.argmax(dim=-1)
        # mix_ids = torch.cat([ori_text,noise_text,mlm_text],dim=0)
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            image_feats, mid_txext_feats = self.base_model(images, ori_text,"mode1")
        xx, (qs, ks, vs), attns, atten_outs = self.clip_encode_text_dense(mid_txext_feats,eos_position, n=8)
        t_feats_ori = F.normalize(xx, dim=-1)
        # t_feats_ori = xx.float()

        i_feats = image_feats[:, 0, :].float()
        cosine = ((F.normalize(image_feats[:,0,:],dim=-1)) @ t_feats_ori.T)
        grad_emap = self.grad_eclip_enhanced_batched(cosine, qs, ks, vs, atten_outs, attns, eos_position,scales=[1, 2])
        # grad_emap = self.grad_eclip(cosine,qs,ks,vs,atten_outs,eos_position)


        noise_text_feature=self.base_model.encode_text(noise_text)
        noise_t_feats = noise_text_feature[torch.arange(noise_text_feature.shape[0]), eos_position].float()

        logit_scale = self.logit_scale

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm_ddp(self.args,i_feats, noise_t_feats, batch['pids'], logit_scale)})
            # ret.update({'sdm_loss':objectives.compute_sdm(i_feats, noise_t_feats, batch['pids'], logit_scale)})
        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'att_mlm' in self.current_task:
            for att_type in ['shoes','hairstyle','genders','top','trousers','belongings']:
                mlm_ids = batch[att_type+'_mlm_ids']

                mlm_feats = self.base_model.encode_text(mlm_ids)

                x = self.cross_former(mlm_feats, image_feats, image_feats)

                x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

                scores = x.float().reshape(-1, self.args.vocab_size)
                mlm_labels = batch[att_type+'_mlm_labels'].reshape(-1)
                ret.update({att_type+'_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

                pred = scores.max(1)[1]
                mlm_label_idx = torch.nonzero(mlm_labels)
                acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                ret.update({att_type+'_acc': acc})

        return ret,grad_emap

def build_model(args, num_classes=11003):
    model = IRRA2(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model

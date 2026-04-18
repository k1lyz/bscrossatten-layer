import torch
import torch.nn.functional as F
import math

# 原版多标签交叉熵损失（保持不变）
def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

# =========================================================================
# 毕设核心创新：拓扑与密度感知的自适应层次对比损失 (TAA-HCL) 
# 消融模式: 
# A: 仅父节点排斥 | B: A + 兄弟节点排斥 | C: B + 密度自适应Margin | D: C + Focal加权
# =========================================================================
def topology_aware_hierarchical_loss(text_features, label_embeddings, value2slot, active_labels, base_margin=0.6, gamma=2.0, ablation_mode='D'):
    loss = 0.0
    valid_triplets = 0
    
    # 采用 Mean + Max 混合池化提取文本证据 (适配无交叉注意力模型)
    text_mean = text_features.mean(dim=1)
    text_max = text_features.max(dim=1)[0]
    text_evidence = (text_mean + text_max) / 2.0 
    
    # 构建父节点到子节点的反向映射表 (用于寻找兄弟节点和计算密度)
    slot2children = {}
    if ablation_mode in ['B', 'C', 'D']:
        for child, parent in value2slot.items():
            if parent != -1:
                if parent not in slot2children:
                    slot2children[parent] = []
                slot2children[parent].append(child)

    for b in range(active_labels.size(0)):
        labels = active_labels[b].nonzero(as_tuple=True)[0]
        
        for child_idx in labels:
            child_idx = child_idx.item()
            parent_idx = value2slot.get(child_idx, -1)
            
            if parent_idx != -1:
                v_c = label_embeddings[child_idx]
                v_p = label_embeddings[parent_idx]
                h_c = text_evidence[b]
                
                # ==== 实验 C & D：节点密度自适应 Margin ====
                current_margin = base_margin
                if ablation_mode in ['C', 'D']:
                    # 密度计算：获取该父节点下挂载的子节点(兄弟)总数
                    sibling_count = len(slot2children.get(parent_idx, []))
                    # 密度越大(局部空间越拥挤)，需要的排斥安全距离越宽
                    # 使用 log 平滑系数，防止超大类别分支导致 Margin 爆炸
                    density_factor = 1.0 + 0.15 * math.log(max(sibling_count, 1))
                    current_margin = base_margin * density_factor
                
                # 计算余弦距离
                dist_evidence = 1.0 - F.cosine_similarity(v_c.unsqueeze(0), h_c.unsqueeze(0))
                dist_parent = 1.0 - F.cosine_similarity(v_c.unsqueeze(0), v_p.unsqueeze(0))
                
                # 纵向(父节点)排斥差距
                diff_parent = dist_evidence - dist_parent + current_margin
                diff_sibling = torch.tensor(0.0, device=text_features.device)
                
                # ==== 实验 B, C, D：横向(兄弟节点)排斥 ====
                if ablation_mode in ['B', 'C', 'D']:
                    siblings = slot2children.get(parent_idx, [])
                    negative_siblings = [s for s in siblings if s != child_idx and not active_labels[b][s]]
                    if negative_siblings:
                        v_sibs = label_embeddings[negative_siblings]
                        dist_sibs = 1.0 - F.cosine_similarity(v_c.unsqueeze(0), v_sibs).mean()
                        # 兄弟排斥边界略小于父子边界
                        diff_sibling = dist_evidence - dist_sibs + (current_margin * 0.8)

                # 合并违规程度 (只惩罚大于0的部分)
                total_diff = F.relu(diff_parent)
                if ablation_mode in ['B', 'C', 'D']:
                    total_diff += 0.5 * F.relu(diff_sibling)
                
                if total_diff > 0:
                    # ==== 实验 D (Ours)：Focal 困难样本加权 ====
                    if ablation_mode == 'D':
                        # 如果差异极大（难样本），赋予更高的惩罚权重
                        weight = (total_diff.detach() / current_margin) ** gamma
                        weight = torch.clamp(weight, min=0.1, max=5.0)
                    else:
                        weight = 1.0 # 实验 A, B, C 不加权
                        
                    loss += weight * total_diff
                    valid_triplets += 1
                
    if valid_triplets > 0:
        return (loss / valid_triplets).squeeze()
    return torch.tensor(0.0, device=text_features.device)

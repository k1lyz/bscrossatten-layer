import torch
import torch.nn.functional as F

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


def advanced_hierarchical_constraint_loss(logits, value2slot, depth_dict, device, margin=0.5, base_weight=2.0):
    """
    高阶层次约束损失：支持动态安全间隔 (Margin) 与 深度自适应权重 (Depth-adaptive)
    """
    child_idx = []
    parent_idx = []
    depth_weights = []
    
    for child, parent in value2slot.items():
        if parent != -1 and child < logits.size(1) and parent < logits.size(1):
            child_idx.append(child)
            parent_idx.append(parent)
            
            # 计算深度权重：层级越浅，权重越大
            child_depth = depth_dict.get(child, 2) 
            weight = base_weight ** (2 - child_depth) 
            depth_weights.append(weight)
            
    if not child_idx:
        return torch.tensor(0.0, device=device)

    # GPU 张量化加速
    child_idx = torch.tensor(child_idx, device=device)
    parent_idx = torch.tensor(parent_idx, device=device)
    depth_weights = torch.tensor(depth_weights, device=device, dtype=torch.float32)

    child_logits = logits[:, child_idx]
    parent_logits = logits[:, parent_idx]

    # Margin 边界惩罚与平方平滑
    violation = child_logits - parent_logits + margin
    squared_penalty = violation.clamp(min=0) ** 2
    weighted_penalty = squared_penalty * depth_weights.unsqueeze(0)
    
    return weighted_penalty.mean()

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

def hierarchical_separation_loss(text_features, label_embeddings, value2slot, active_labels, margin=0.5):
    """
    改进一：三元组层次分离损失 (Hierarchical Separation Loss)
    确保子标签嵌入靠近支持它的文本特征，而与父标签保持距离，防止特征坍缩。
    """
    loss = 0.0
    valid_triplets = 0
    
    # 提取池化后的文本证据特征 (对序列长度求平均)
    text_evidence = text_features.mean(dim=1) # [batch_size, hidden_size]
    
    for b in range(active_labels.size(0)):
        labels = active_labels[b].nonzero(as_tuple=True)[0]
        for child_idx in labels:
            child_idx = child_idx.item()
            parent_idx = value2slot.get(child_idx, -1)
            
            if parent_idx != -1:
                v_c = label_embeddings[child_idx]
                v_p = label_embeddings[parent_idx]
                h_c = text_evidence[b]
                
                # 计算余弦距离: D(v_c, h_c) - D(v_c, v_p) + margin
                dist_evidence = 1.0 - F.cosine_similarity(v_c.unsqueeze(0), h_c.unsqueeze(0))
                dist_parent = 1.0 - F.cosine_similarity(v_c.unsqueeze(0), v_p.unsqueeze(0))
                
                triplet_loss = F.relu(dist_evidence - dist_parent + margin)
                loss += triplet_loss
                valid_triplets += 1
                
    if valid_triplets > 0:
        return (loss / valid_triplets).squeeze()
    return torch.tensor(0.0, device=text_features.device)

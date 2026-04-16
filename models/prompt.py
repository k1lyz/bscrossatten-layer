from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy, hierarchical_separation_loss
from .graph import GraphEncoder
from .attention import CrossAttention
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    def forward(self, x):
        x = F.embedding(x, self.weight(), self.padding_idx)
        return x


class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, value2slot=None, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        
        # === 消融实验开关与拓扑关系 ===
        self.value2slot = value2slot
        self.ablation_logits_mask = kwargs.get('ablation_logits_mask', False)
        self.ablation_hierarchical_loss = kwargs.get('ablation_hierarchical_loss', False)
        self.ablation_cross_attn = kwargs.get('ablation_cross_attn', False)
        self.ablation_deep_prefix = kwargs.get('ablation_deep_prefix', False)
        
        # 改进三：文本-标签概念的深度交叉注意力融合模块
        if self.ablation_cross_attn:
            self.cross_attention = CrossAttention(
                embed_dim=config.hidden_size, 
                num_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob
            )

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'), weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        # label prompt mask
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
        self.set_input_embeddings(embedding)

        decoder = self.get_output_embeddings()
        self.vocab_size = decoder.bias.size(0)
        decoder.bias = nn.Parameter(nn.functional.pad(
            decoder.bias.data,
            (0, embedding.size - decoder.bias.shape[0]),
            "constant",
            0,
        ))

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 动态获取当前 GPU Replica 上的 GraphEmbedding 并产生最新的张量权重
        hidden_states = self.cls.predictions.transform(sequence_output)
        weight = self.get_input_embeddings().raw_weight()
        bias = self.get_output_embeddings().bias

        # =====================================================================
        # 改进三核心：显式的多维语义交互 (Cross-Attention)
        # =====================================================================
        if self.ablation_cross_attn:
            # 提取属于标签的那部分特征
            label_weights = weight[self.vocab_size : self.vocab_size + self.num_labels]
            bsz = hidden_states.size(0)
            expanded_label_weights = label_weights.unsqueeze(0).expand(bsz, -1, -1)
            
            # Query: 文本序列特征; Key/Value: 标签概念特征
            attn_output, _, _ = self.cross_attention(
                hidden_states=hidden_states, 
                key_value_states=expanded_label_weights
            )
            # 残差融合，文本吸收了强相关的标签概念
            hidden_states = hidden_states + attn_output

        prediction_scores = F.linear(hidden_states, weight, bias)
        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))
            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1, prediction_scores.size(-1))
            multiclass_logits = multiclass_logits[:, self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            masked_lm_loss += multiclass_loss

            # =====================================================================
            # 改进一核心(Train)：注入三元组层次分离损失，并兼容多卡 DataParallel
            # =====================================================================
            if self.ablation_hierarchical_loss and self.value2slot is not None:
                # 获取当前 GPU 上被 DataParallel 分配的实际 batch_size (如 8)
                bsz = sequence_output.size(0)
                
                # 将展平的 labels 恢复为 [bsz, max_depth, num_labels]
                # 然后在深度维度 (dim=1) 上用 .any() 压缩，得到当前样本真正激活的类 [bsz, num_labels]
                active_labels = (labels.view(bsz, -1, self.num_labels) == 1).any(dim=1)
                
                label_weights = weight[self.vocab_size : self.vocab_size + self.num_labels]
                sep_loss = hierarchical_separation_loss(
                    text_features=sequence_output, 
                    label_embeddings=label_weights,
                    value2slot=self.value2slot,
                    active_labels=active_labels,
                    margin=0.5
                )
                masked_lm_loss += 0.1 * sep_loss  # 平衡系数

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        ret = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1, prediction_scores.size(-1))
        
        prediction_scores = prediction_scores[:, self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            activated_nodes = set() # 记录已激活的节点
            
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    # =====================================================================
                    # 改进一核心(Test)：基于对数几率掩码的受限解码 (Constrained Decoding)
                    # =====================================================================
                    if self.ablation_logits_mask and self.value2slot is not None:
                        parent_id = self.value2slot.get(l, -1)
                        # 如果存在父节点，且父节点未被激活，强行将其 Logits 拉至负无穷
                        if parent_id != -1 and parent_id not in activated_nodes:
                            score[l] = -float('inf')
                            
                    if score[l] > 0:
                        predict_labels[-1].append(l)
                        activated_nodes.add(l) # 注册激活状态
        return predict_labels, prediction_scores

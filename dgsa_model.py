# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
import math
from BERT import BertModel, BertPreTrainedModel

try:
    import apex
    #apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    #apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    BertLayerNorm = apex.normalization.FusedLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class DiGCNLayerAtt(nn.Module):
    def __init__(self, hidden_size, use_weight=False):
        super(DiGCNLayerAtt, self).__init__()
        self.temper = hidden_size ** 0.5
        self.use_weight = use_weight
        self.relu = nn.ReLU()
        self.relu_left = nn.ReLU()
        self.relu_self = nn.ReLU()
        self.relu_right = nn.ReLU()

        self.linear = nn.Linear(hidden_size, hidden_size)

        self.left_linear = nn.Linear(hidden_size, hidden_size)
        self.right_linear = nn.Linear(hidden_size, hidden_size)
        self.self_linear = nn.Linear(hidden_size, hidden_size)

        self.output_layer_norm = BertLayerNorm(hidden_size)

        self.reset_parameters(self.linear)
        self.reset_parameters(self.left_linear)
        self.reset_parameters(self.right_linear)
        self.reset_parameters(self.self_linear)

        self.softmax = nn.Softmax(dim=-1)

    def reset_parameters(self, linear):
        init.xavier_normal_(linear.weight)
        # init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(linear.bias, -bound, bound)

    def get_att(self, matrix_1, matrix_2, adjacency_matrix):
        u = torch.matmul(matrix_1.float(), matrix_2.permute(0, 2, 1).float()) / self.temper
        attention_scores = self.softmax(u)
        delta_exp_u = torch.mul(attention_scores, adjacency_matrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10).type_as(matrix_1)
        return attention

    def forward(self, hidden_state, adjacency_matrix, output_attention=False):
        context_attention = self.get_att(hidden_state, hidden_state, adjacency_matrix)

        hidden_state_left = self.left_linear(hidden_state)
        hidden_state_self = self.self_linear(hidden_state)
        hidden_state_right = self.right_linear(hidden_state)

        context_attention_left = torch.triu(context_attention, diagonal=1)
        context_attention_self = torch.triu(context_attention, diagonal=0) - context_attention_left
        context_attention_right = context_attention - torch.triu(context_attention, diagonal=0)

        context_attention = torch.bmm(context_attention_left.float(), hidden_state_left.float()) \
                            + torch.bmm(context_attention_self.float(), hidden_state_self.float()) \
                            + torch.bmm(context_attention_right.float(), hidden_state_right.float())

        output_attention_list = [context_attention_left, context_attention_self, context_attention_right]


        o = self.output_layer_norm(context_attention.type_as(hidden_state))
        output = self.relu(o).type_as(hidden_state)

        if output_attention is True:
            return (output, output_attention_list)
        return output

class DiGCNModuleAtt(nn.Module):
    def __init__(self, layer_number, hidden_size, use_weight=False, output_all_layers=False):
        super(DiGCNModuleAtt, self).__init__()
        if layer_number < 1:
            raise ValueError()
        self.layer_number = layer_number
        self.output_all_layers = output_all_layers
        self.GCNLayers = nn.ModuleList(([DiGCNLayerAtt(hidden_size, use_weight)
                                         for _ in range(self.layer_number)]))

    def forward(self, hidden_state, adjacency_matrix, output_attention=False):
        # hidden_state = self.first_GCNLayer(hidden_state, adjacency_matrix, type_seq, type_matrix)
        # all_output_layers.append(hidden_state)

        all_output_layers = []

        output_attention_list = []
        for gcn in self.GCNLayers:
            hidden_state = gcn(hidden_state, adjacency_matrix, output_attention=output_attention)
            if output_attention is True:
                hidden_state, output_attention_list = hidden_state
            all_output_layers.append(hidden_state)

        if self.output_all_layers:
            if output_attention is True:
                return all_output_layers, output_attention_list
            return all_output_layers
        else:
            if output_attention is True:
                return all_output_layers[-1], output_attention_list
            return all_output_layers[-1]

class DGSA(BertPreTrainedModel):
    def __init__(self, config, num_labels, type_num, gcn_layer_number=3, use_weight=False):
        super(DGSA, self).__init__(config)
        self.num_labels = num_labels
        self.type_num = type_num
        self.gcn_layer_number = gcn_layer_number
        self.bert = BertModel(config)
        self.gcn = DiGCNModuleAtt(gcn_layer_number, config.hidden_size, use_weight=use_weight, output_all_layers=False)
        self.dep_embedding = nn.Embedding(type_num, config.hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, adjacency_matrix, token_type_ids=None, attention_mask=None, labels=None,
                valid_ids=None, output_attention=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if valid_ids is not None:
            batch_size, max_len, feat_dim = sequence_output.shape
            valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
            for i in range(batch_size):
                temp = sequence_output[i][valid_ids[i] == 1]
                valid_output[i][:temp.size(0)] = temp
        else:
            valid_output = sequence_output
        sequence_output = self.dropout(valid_output)

        sequence_output = self.gcn(sequence_output, adjacency_matrix, output_attention=output_attention)
        if output_attention is True:
            sequence_output, output_attention_list = sequence_output
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if output_attention is True:
                return loss, output_attention_list
            return loss
        else:
            if output_attention is True:
                return logits, output_attention_list
            return logits
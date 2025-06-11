import torch
import torch.nn as nn
import numpy as np

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_head,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps
    ):
        super(MultiheadAttention, self).__init__()
        if hidden_size % n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_head)
            )
        self.num_attention_heads = n_head
        self.attention_head_size = int(hidden_size / n_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = np.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x
    
    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)

        return hidden_states   
    
class SelfAttentionSessionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_head,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps
    ):
        super(SelfAttentionSessionEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        self.self_attention = MultiheadAttention(
            n_head,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )

        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_tensor, attention_mask, padding_mask):
        input_tensor = self.LayerNorm(input_tensor)
        self_attention_output = self.self_attention(input_tensor, attention_mask)
        self_attention_output = input_tensor + self_attention_output
        self_attention_output = self.LayerNorm(self_attention_output)
        feedforward_output = self.feedforward(self_attention_output)
        feedforward_output = input_tensor + feedforward_output
        output = feedforward_output * padding_mask
        return output

from module import *

class SelfAttentiveSessionEncoder(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_size = 50,
        n_head = 1,
        hidden_dropout_prob = 0.5,
        attn_dropout_prob = 0.5,
        layer_norm_eps = 0.00001,
        n_layers = 2,
        max_session_length = 10
    ):
        super(SelfAttentiveSessionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.n_layers = n_layers
        self.max_session_length = max_session_length

        # define layers
        self.item_embedding = nn.Embedding(num_items, hidden_size, max_norm=1, padding_idx=0)
        self.indices = nn.Parameter(
            torch.arange(num_items, dtype=torch.long), requires_grad=False
        )
        self.position_embedding = nn.Embedding(self.max_session_length, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layers = nn.ModuleList([
            SelfAttentionSessionEncoderLayer(
                hidden_size,
                n_head,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps
            ) for _ in range(n_layers)
        ])

    def get_attention_mask(self,item_seq, bidirectional=False):
        # Create attention mask to achieve future blinding
        attention_mask = (item_seq != 0).unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            attention_mask = torch.tril(attention_mask.expand(-1, -1, item_seq.size(-1), -1))
        attention_mask = torch.where(attention_mask == 0, torch.tensor(-1e9), torch.tensor(0.0))
        return attention_mask
    
    def get_padding_mask(self, item_seq):
        # Create padding mask to ignore padding items in the sequence
        # "mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)" in SASRec source code
        padding_mask = (item_seq != 0).float().unsqueeze(-1)
        return padding_mask

    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_embedding = self.item_embedding(item_seq)
        input_embedding = item_embedding + position_embedding
        input_embedding = self.dropout(input_embedding)

        padding_mask = self.get_padding_mask(item_seq)
        input_embedding = input_embedding * padding_mask

        attention_mask = self.get_attention_mask(item_seq)
        for layer in self.layers:
            input_embedding = layer(input_embedding, attention_mask, padding_mask)
        
        output_embedding = self.layer_norm(input_embedding)

        return output_embedding
    
    def get_logits(self, batch):
        # given a bacth of sequences, return logits for the next item of each sequence
        session_embeddings = self.forward(batch)
        # dimension of session_embeddings: [batch_size, max_session_length, hidden_size]
        # take the last item representation as the session representation
        session_embeddings = session_embeddings[:, -1, :]
        logits = session_embeddings @ self.item_embedding(self.indices).t()
        return logits

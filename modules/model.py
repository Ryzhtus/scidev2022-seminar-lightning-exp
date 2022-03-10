import torch
import torch.nn as nn

class NERTagger(nn.Module):
    def __init__(self, output_dim: int, embedding_matrix: torch.FloatTensor,
                 feedforward_dim=100, dropout_rate=0.1):
        super(NERTagger, self).__init__()
        embedding_dim = embedding_matrix.size(-1)
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.feedforward = nn.Linear(embedding_dim, feedforward_dim)
        self.out = nn.Linear(feedforward_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.LongTensor):
        x = self.embeddings(x)
        x = self.feedforward(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.out(x)
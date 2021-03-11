import torch
from torch import nn
from torch.nn import functional as F

class BertEmbed_FactCheckAttention(nn.Module):
  """
  Pretrained BERT embedding + KnowledgeBase docs embeddings
  wrapped in attention mechanism
  """
  
  def __init__(self, n_classes, n_doc, embed_size, dropout=0.1):
    """
    :param n_classes: Number of classes
    :param n_doc: Number of documents in knowledge base
    :param embed_size: Embedding size
    :param dropout: Dropout rate
    """
    super(BertEmbed_FactCheckAttention, self).__init__()
    self.n_classes = n_classes
    self.n_doc = n_doc
    self.embed_size = embed_size
    self.dropout = nn.Dropout(dropout)
    
    self.attn_proj = nn.Linear(
      in_features=embed_size,
      out_features=embed_size,
      bias=False
    )
    self.linear1 = nn.Linear(in_features=2*embed_size, out_features=embed_size)
    self.linear2 = nn.Linear(in_features=embed_size, out_features=n_classes)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, tweet_embeddings, kb_embeddings):
    """
    :param tweet_embeddings: batch_size * embed_size
    :param kb_embeddings: n_doc * embed_size
    :return: batch_size * n_classes
    """
    # n_doc * embed_size
    attn_projection = self.attn_proj(kb_embeddings)
    
    # batch_size * n_doc
    attn_scores = torch.matmul(tweet_embeddings, torch.transpose(attn_projection, 0, 1))
    attn_dist = self.softmax(attn_scores)
    
    # batch_size * embed_size
    attention_out = torch.matmul(attn_dist, kb_embeddings)
    
    linear_out = self.dropout(self.linear1(
      torch.hstack((tweet_embeddings, attention_out))
    ))
    
    return self.softmax(self.linear2(F.relu(linear_out)))

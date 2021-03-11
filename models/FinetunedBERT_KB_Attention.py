import torch
from torch import nn
from torch.nn import functional as F
import transformers

class FactCheckAttentionNetwork(nn.Module):
  def __init__(self, n_classes, n_doc, dropout=0.1, bert_variant='bert-base-cased'):
    """
    :param n_classes: Number of classes
    :param n_doc: Number of documents in knowledge base
    :param dropout: Dropout rate
    :param bert_variant: BERT variant as described in HuggingFace transformers module
    """
    super(FactCheckAttentionNetwork, self).__init__()
    self.n_classes = n_classes
    self.n_doc = n_doc
    self.bert_tweet = transformers.BertModel.from_pretrained(bert_variant)
    self.embed_size = self.bert_tweet.config.hidden_size
    self.attn_proj = nn.Linear(
      in_features=self.embed_size,
      out_features=self.embed_size,
      bias=False
    )
    self.softmax_kb = nn.Softmax(dim=1)
    self.dropout = nn.Dropout(dropout)
    self.linear1 = nn.Linear(in_features=self.embed_size * 2, out_features=self.embed_size)
    self.linear2 = nn.Linear(in_features=self.embed_size, out_features=n_classes)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, input_ids, attn_mask, kb_embeddings):
    """
    :param input_ids: Input IDs from BERT Tokenizer class. batch_size * MAX_LEN
    :param attn_mask: Attention mask from BERT Tokenizer class. batch_size * MAX_LEN
    :param kb_embeddings: n_doc * embed_size
    :return: batch_size * n_classes
    """
    
    # n_doc * embed_size
    kb_projection = self.attn_proj(kb_embeddings)
    
    # batch_size * embed_size
    tweet_embed = self.bert_tweet(
      input_ids=input_ids,
      attention_mask=attn_mask
    ).pooler_output
    
    # batch_size * n_doc
    attn_scores = torch.matmul(self.dropout(tweet_embed), torch.transpose(kb_projection, 0, 1))
    
    # batch_size * n_doc
    attn_dist = self.softmax_kb(attn_scores)
    
    # batch_size * embed_size
    attn_output = torch.matmul(attn_dist, kb_embeddings)
    
    # batch_size * (2 x embed_size)
    input_for_ff = self.dropout(torch.hstack((tweet_embed, attn_output)))
    
    out = self.linear2(F.relu(self.linear1(input_for_ff)))
    
    # batch_size * n_classes
    return self.softmax(out)

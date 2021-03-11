from torch import nn
from torch.nn import functional as F
import transformers

class BERT_Finetuning(nn.Module):
  def __init__(self, n_classes, dropout=0.1, bert_variant='bert-base-cased'):
    """
    Finetuning of BERT using a feed-forward layer.
    :param n_classes: Number of classes
    :param dropout: Dropout rate
    :param bert_variant: BERT variant as described in HuggingFace transformers module
    """
    super(BERT_Finetuning, self).__init__()

    self.n_classes = n_classes
    self.bert = transformers.BertModel.from_pretrained(bert_variant)
    self.linear1 = nn.Linear(in_features=self.bert.config.hidden_size, out_features=self.bert.config.hidden_size)
    self.linear2 = nn.Linear(in_features=self.bert.config.hidden_size, out_features=n_classes)
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
    """
    :param input_ids: Input IDs from BERT Tokenizer class
    :param attention_mask: Attention mask from BERT Tokenizer class
    :return: batch_size * n_classes
    """
    pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).pooler_output

    out = self.linear1(self.dropout(pooled_output))

    return self.softmax(self.linear2(F.relu(out)))

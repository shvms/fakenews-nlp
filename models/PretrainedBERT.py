from torch import nn

"""
Extra baseline
"""
class BaselineBERT_NN(nn.Module):
  """
  Pretrained BERT embedding + Feedforward network
  """
  
  def __init__(self, n_classes, embed_size):
    super(BaselineBERT_NN, self).__init__()
    self.n_classes = n_classes
    self.embed_size = embed_size    # BERT embedding size
    self.linear = nn.Linear(in_features=embed_size, out_features=n_classes, bias=False)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, embedding):
    out = self.linear(embedding)
    return self.softmax(out)

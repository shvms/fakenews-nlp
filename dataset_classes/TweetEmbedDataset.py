from torch.utils import data

class TweetEmbedDataset(data.Dataset):
  """
  Dataset with BERT embedding and label as columns.
  """
  
  def __init__(self, embedding, label):
    self.embedding = embedding
    self.label = label
  
  def __len__(self):
    return len(self.embedding)
  
  def __getitem__(self, idx):
    return {
      'embedding': self.embedding.iloc[idx].flatten(),
      'label': self.label.iloc[idx]
    }

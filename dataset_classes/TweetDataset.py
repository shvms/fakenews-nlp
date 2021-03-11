from torch.utils import data

class TweetDataset(data.Dataset):
  """
  Tweets dataset with encoded tweet and label.
  """
  
  def __init__(self, tweets, labels, tokenizer, max_length):
    self.tweets = tweets
    self.labels = labels
    self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer)
    self.max_length = max_length
  
  def __len__(self):
    return len(self.tweets)
  
  def __getitem__(self, idx):
    tweet = self.tweets.iloc[idx]
    label = self.labels.iloc[idx]
    
    encoding = self.tokenizer.encode_plus(
      tweet,
      max_length=self.max_length,
      add_special_tokens=True,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_token_type_ids=False,
      return_tensors='pt'
    )
    
    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'label': label
    }

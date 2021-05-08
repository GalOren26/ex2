import torch.nn as nn
from torch.autograd import Variable
import torch.tensor as tensor 
class Rnn(nn.Module):
  """Simple LSMT-based language model"""
  
  #to change embbeding dim 
  #to change drop out  val 
   # change init_hidden 
   # check the lerning rate
   #change  dp_keep_prob
   
   # consider of we need clip_grad_norm
   #consider using step insted 
   
   #CHECK FOR PREPLXTIY CALC 
   
  def __init__(self,net,embedding_dim, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob,lr=1,lr_decay_base=2,ephocs_witout_decay=4):
    super(Rnn, self).__init__()
    self.embedding_dim = embedding_dim
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.net_name=net
    self.first_lr=lr
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    if net=="lstm":
        self.net = nn.LSTM(input_size=embedding_dim,
                                hidden_size=embedding_dim,
                                num_layers=num_layers,
                                dropout=1 - dp_keep_prob)
    
    else:
        self.net =nn.GRU(input_size=embedding_dim,
                                hidden_size=embedding_dim,
                                num_layers=num_layers,
                                dropout=1 - dp_keep_prob)
    self.sm_fc = nn.Linear(in_features=embedding_dim,
                           out_features=vocab_size)
    self.lr=lr
    self.ephocs_witout_decay=ephocs_witout_decay
    self.lr_decay_base=lr_decay_base
    self.init_weights()
    self.train_perplexity=[]
    self.test_perplexity=[]
    self.already_decayed=False
  def init_weights(self):
    init_range = 0.1
    nn.init.xavier_normal_(self.word_embeddings.weight.data)
    # self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    nn.init.xavier_normal_(self.sm_fc.weight.data)
    
    # self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data
    if self.net_name=="lstm":
        return (Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()),
            Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()))
    else :
        return   Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_())

  def forward(self, inputs, hidden):
    if self.dp_keep_prob==1:
      embeds=self.word_embeddings(inputs)
    else : 
      embeds = self.dropout(self.word_embeddings(inputs))
    net_out, hidden = self.net(embeds, hidden)
    if self.dp_keep_prob!=1:
       net_out = self.dropout(net_out)
    logits = self.sm_fc(net_out.view(-1, self.embedding_dim))
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if type(h) is  not tuple:
    return Variable(h.data)
  # else:
  #  return tuple(repackage_hidden(v) for v in h)
  else:
      temp=[]
      for i in h:
          temp.append(Variable(i.data))
      return tuple(temp)

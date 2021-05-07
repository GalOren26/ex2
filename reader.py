
import collections
import os

import numpy as np
def read_words(filename):
  with open(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
  data = read_words(filename)
  words=set(data)
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict((v, k) for k, v in word_to_id.items())
  return word_to_id, id_to_word


def file_to_word_ids(filename, word_to_id):
  data = read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def files_raw_data(paths,names,word_to_id,trainRatio=1):
    data={}
    for idx,path in enumerate(paths):
        data[names[idx]]=file_to_word_ids(path,word_to_id)
    if trainRatio!=1: 
        data['train']=RandomSampleAcordionRatio(data['train'],trainRatio)
    return data     
def RandomSampleAcordionRatio(data,ratio):
     start_idx=int(np.random.uniform(0,1-ratio)*len(data))
     end_idx=start_idx+int(ratio*len(data))
     return data[start_idx:end_idx]
def full_path_files(base,names_of_files):
    return [base+name for name in names_of_files]

def ptb_iterator(raw_data, batch_size, num_steps):
       
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers. 
  
  make the raw data iterater in dim  [batch_size,seq_len] when all the data store at 
  [batch_size,batch_len]-> batch_len*batch*size=len(data)
  
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]


  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

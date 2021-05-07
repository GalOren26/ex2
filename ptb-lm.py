
import optuna
import time
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import torch.tensor as tensor 
import numpy as np
# from  reader import * 
# from lm import * 
import logging
from lm import *
from reader import * 

# to change hidden unit to 200 
# args={"load":"","trainRatio":1,"data":"data/","hidden_size":200 ,"seq_len":20,"num_layers":2,"batch_size":20,"num_epochs":13,"dp_keep_prob":1,"inital_lr":25,"save":"lm_model.pt","net":"lstm"}


init_args={"load":"","trainRatio":1,"data":"data/","log_path":"./log_gru_DO.txt"}
def save_checkpoint(model,save_path, epoch,trained):
    torch.save({
        'model': model,
        'epoch': epoch
        'trained':trained
    }, save_path)
def load_checkpoint(model,load_path):
    
    checkpoint = torch.load(load_path)
    model=checkpoint[]
    epoch = checkpoint['epoch']
    if checkpoint['trained']:
        print('continue to train from epoch {}, see log file for history :)\n'.format(epoch))
    else:
        print('load model from file! test on test data :)\n')
    return model,epoch 

    
def run_epoch(model, data, is_train=False, lr=1.0):
  """Runs the model on the given data."""
  if is_train:
    model.train()
  else:
    model.eval()
    
  num_of_seq = ((len(data) // model.batch_size) - 1) // model.seq_len
  start_time = time.time()
  hidden = model.init_hidden()
  costs = 0.0
  iters = 0
  for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
    inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).to(device)
    model.zero_grad()
    hidden = repackage_hidden(hidden)
    outputs, hidden = model(inputs, hidden)
    targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).to(device)
    tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

    loss = criterion(outputs.view(-1, model.vocab_size), tt)
    costs += loss.item() * model.seq_len
    iters += model.seq_len

    if is_train:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
      for p in model.parameters():
        p.data.add_(-lr, p.grad.data)
      # if step % (num_of_seq // 10) == 10:
      #   print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / num_of_seq, np.exp(costs / iters),
      #                                                  iters * model.batch_size / (time.time() - start_time)))
  return np.exp(costs / iters)    









def objective(trial):

    args={"seq_len":trial.suggest_int("seq_len",10,30),"batch_size":trial.suggest_int("batch_size",15,25),"inital_lr":trial.suggest_int("lr",7,20),"num_epochs":trial.suggest_int("num_epocs",10,21), 
          "lr_decay_base"=trial.suggest_float("decay",1,2),"ephocs_witout_decay"=4}
          "load":"","trainRatio":1,"data":"data/", "hidden_size":200 ,"num_layers":2, "dp_keep_prob":trial.suggest_float("droop",0.1,0.9), "save":"lm_model2.pt", "net":"gru ","log_path": init_args["log_path"]}
   

  
    #load model or create it     
    if args["load"]!="":
        model,epoch_num=load_checkpoint(model,args['load'])
    else:      
        epoch_num=args["num_epochs"]
        model = Rnn(net=args["net"],embedding_dim=trial.suggest_int("embbeding",150,300),seq_len=args["seq_len"], batch_size=args["batch_size"],
                 vocab_size=vocab_size, num_layers=args["num_layers"], dp_keep_prob=args["dp_keep_prob"],lr=args["inital_lr"],lr_decay_base=args["lr_decay_base"],ephocs_witout_decay=args["ephocs_witout_decay"])
     #convert to device 
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     model.to(device)
    # if load is avilable get lr and num of ephoc else take the defualt. 
    print(trial.params.items()) #CHANGE TO model.paramters()
    print("-------Training------")
    
    prep_valid_list=[]
    f = open(args["log_path"], "a")
    for epoch in range(epoch_num):
      if ( epoch>model.ephocs_witout_decay):
            model.lr=model.lr/model.lr_decay_base
      train_perplexity = run_epoch(model, data["train"], True, model.lr)
      valid_perplexity=run_epoch(model, data["valid"])

      if len(prep_valid_list)<5:
        prep_valid_list.append(valid_perplexity)
            
      elif valid_perplexity>max(prep_valid_list) :
            min_val=min(prep_valid_list)
            print("exit from condtion")  # to change 
            f.write('Validation perplexity at best epoch  {}'.format(min_val))
            f.write('Validation perplexity at last ephoc  {}: {:8.2f}'.format(epoch,valid_perplexity))
            f.write("".join(["    {}: {}".format(key, value) for key, value in trial.params.items()])) # to change later 
            f.write("\n\n")
            return valid_perplexity
      else:
          max_key=max(prep_valid_list)
          prep_valid_list[prep_valid_list.index(max_key)]=valid_perplexity
          if min(prep_valid_list)==valid_perplexity:
              print("save model new result :)\n")
              save_checkpoint(model,args["save"],epoch,True)
          avg=sum(prep_valid_list)/len(prep_valid_list)
          bool2=100*(abs((valid_perplexity-avg))/avg)<2
          if bool2:
            model.lr*=model.lr_decay_base*2
            print("true") #cahnge
            f.write("true\n") # change 
        
      print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_perplexity))
      print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, valid_perplexity))
      f.write('Train train train train train train  perplexity at epoch {}: {:8.2f}\n'.format(epoch, train_perplexity))
      f.write('Validation perplexity at epoch {}: {:8.2f}\n\n'.format(epoch, valid_perplexity))
      trial.report(valid_perplexity, epoch)
        # Handle pruning based on the intermediate value.
        
        
        
      if trial.should_prune():  #to change 
            print("exit from shold prone")
            f.write('Train should_prune  perplexity at epoch {}: {:8.2f}\n'.format(epoch, train_perplexity))
            f.write('Validation perplexity at epoch {}: {:8.2f}\n\n'.format(epoch, valid_perplexity))
            raise optuna.exceptions.TrialPruned()
    min_val=min(prep_valid_list)
    f.write('Validation perplexity at best epoch {}'.format(min_val))
    f.write(("".join(["    {}: {}".format(key, value) for key, value in trial.params.items()])))
    f.write("\n\n")
    f.close()
    return valid_perplexity
   

if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #open log file resdults 
    f = open(init_args["log_path"], "w")
    f.write("hello from traing with Optuna\n")
    f.close()
    ## --read data and parse to voacb 
    paths=["ptb.train.txt","ptb.valid.txt","ptb.test.txt"]
    names_of_files=["train","valid","test"]
    paths = full_path_files(init_args["data"],paths)
    train_name=paths[0]
    word_to_id, id_2_word =build_vocab(train_name)
    # return for each file list of the words in the file encoded to id by the word to id dict 
    data=files_raw_data(paths,names_of_files,word_to_id,init_args["trainRatio"])
    vocab_size = len(word_to_id)
    criterion = nn.CrossEntropyLoss()  
    
    # print('Vocabluary size: {}'.format(vocab_size))
     # print("---  Testing ---")
    # model.batch_size = 1 # to make sure we process all the data
    
    # print('Test Perplexity: {:8.2f}'.format(run_epoch(model,  data["test"])))
    # with open(args["save"], 'wb') as f:
    #     save_checkpoint(model,self.lr,self.lr_decay_base, self. ephocs_witout_decay, f,cur_epoch)
    # print("----Done! ---")
   
    # op params with optuna 
    
    study = optuna.create_study( direction='minimize', pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=7, interval_steps=1)
    )
    study.optimize(objective, n_trials=70)
    plot_optimization_history(study)
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

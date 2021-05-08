# ex2

##intro : 
In this project we implemented ex2 as part of deep learning course in tau.
our goal is to predict the next word using the  Tree Bank data set.
Instead of accuracy, we measure the performance by perplexity, which is quite similar to cross-entropy loss done in classification based
NN.
in order to improve performance we used  apply drop out for the net according the decription in Recurrent Neural
Network Regularization", by Zaremba meaning only between the layers and on input of first layer and output of last layer but not between the steps of each layer
in order to preserve the memory capabilty.

### How to run :
#### launch the project 
  there is sevreal way to run this project :
  1.the easist one is simply enter this link https://colab.research.google.com/drive/1S8K23e0BXSqZnnAWQeIqVf3P4uTUwPP6?usp=sharing
  open a colab project and type ctrl+F9 to run all cell toghter then it automaticaly clone files from this project and start the exrice .
  2. import the appended ex2.iptnb at the repo to colab by  and then do what described at 1 .
  3. clone the repo and run loccaly by run the ptb-lm.py file in editor like spyder or simalar make sure that you have all the dependencies istalled (e.g pytorch ,matplotlib ext)
  if not just install it from pip ;)
  #### chose configuration and change paramaters 
  after launch the project you will get a simple ui but detailed ask you what you want to do , folow it.
  if you want to launch the project with castum settings change args dict (find it in ptb-lm.py for localy or last cell at colab) ans change it as you wish.
  
  ## additinal features-save,load, log files
  
  The project give thhe ability to perform serialization at any time.
   The execution of the training operation can be interrupted at any time by stopping the run - and the state of the model as well as number of ephocs alresy perform  will        automatically be maintained for later train.
  Also after the end of the training stage  the best model found is saved (at each epoch we test for improvemnt on validtion dataset and save the imroved version of model for later execuation 

  In order to load a desired model, the path on it must be provided in the "load" parameter in the args dictionary, and also in order to save to a desired location, change the  "save" arg in the args dictionary in code.
  By default for each possible configuration out of the 4 a file with the name of the option in the folder is automatically saved so that the default values can be used without having to change them.

  In addition, a log file is saved for each configuration out of the 4   that saved the validation and train values for each ephoc of the run .


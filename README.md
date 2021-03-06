# Fake news detection using Ensemble model

### ||   FAKE NEWS DETECTION USING ENSEMBLE MODEL   ||     

### Prerequisites

What things you need to install the software and how to install them:

1. Python 3.6 
   - This setup requires that your machine has python 3.6 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in *how to run software section*). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.  
   - Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic. 
2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/
3. You will also need to download and install below 3 packages after you install either python or anaconda from the steps above
   - Python 3+
   - Keras 2+
   - TensorFlow 1+

  - if you have chosen to install python 3.6 then run below commands in command prompt/terminal to install these packages
   ```
   pip install keras
   pip install tensorflow
   
   ```
   - if you have chosen to install anaconda then run below commands in anaconda prompt to install these packages
   ```
   conda install -c
   conda install -c anaconda 
   conda install -c anaconda 
   ```     

### Block Diagram 

The image bellow shows the process / block diagram of the model 

<p align="center">
  <img width="600" height="750" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/BlockDiagram.jpg">
</p>

### Dataset used

The dataset source used in the project is from Kaggle and the link for the same is :
        https://www.kaggle.com/c/fake-news/data
        
train.csv: A full training dataset with the following attributes:<br />

id: unique id for a news article <br />
title: the title of a news article <br />
author: author of the news article <br />
text: the text of the article; could be incomplete <br />
label: a label that marks the article as potentially unreliable <br />
1: unreliable <br />
0: reliable <br />
test.csv: A testing training dataset with all the same attributes at train.csv without the label. <br />
<br />
The orignal dataset contains 3 files and 11 columns , which include 6 for string , 3 for id , 2 for integer respectively.
The dataset has 3 files in CSV format where only 2 files were used which are "train.CSV" and "test.csv" 

For the Train dataset : 
* Column 1: ID
* Column 2: Title
* Column 3: Author
* Column 4: Text
* Column 5: Label

For the test dataset :
* Column 1: ID
* Column 2: Title
* Column 3: Author
* Column 4: Text

To keep the project simple we have choosen limited number of variable from the orignal dataset for the classification. The other variables can 
be added later to add some more complexity and enhance the features.

#### Train Dataset

<p align="center">
  <img width="700" height="175" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/train.png">
</p>

#### Test Dataset 

<p align="center">
  <img width="700" height="250" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/test.png">
</p>

### Files and File descriptions
 
#### Dataset.ipynb
The file contains the information about dataset and the data frame made used using panadas framework.

#### CNN.ipynb
The file contains the code for Convolutional Neural Network CNN model and the approach that has been used to get the desired result.

#### LSTM.ipynb
The file contains the code for Long Short Term Memory LSTM model , and the desired results are derived from it.
### CNN ( Convolutional Neural Networks )

A Convolutional Neural Network is a Deep Learning algorithm which can take in an input and assign importance (learnable weights and biases) to various aspects/objects in the input and be able to differentiate one from the other via labeling them. The pre-processing required in CNN is much lesser as compared to other classification algorithms. A CNN typically has three layers: a convolutional layer, pooling layer, and fully connected layer.
The typical process for an CNN architecture goes with the input being given into convolution layer. The main objective of convolution is to extract features from the input with featured kernel/filters. This is further applied to pooling layer to reduce the dimensions of data. Afterwards Addition of these layers multiple times is carried out and flattend. The flattened output is fed to a feed-forward neural network and backpropagation is applied to every iteration of training. Over a series of epochs, the model can distinguish between dominating and certain low-level features in input and classify them.

For Simple CNN : 
#### Graph
<p align="center">
  <img width="400" height="550" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/simplecnn.png">
</p>

For Convolutional approach : 
#### Graph
<p align="center">
  <img width="400" height="550" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/convocnn.png">
</p>

#### Accuracy and Confusion Matrix 
Output of both the models 
<p align="center">
  <img width="400" height="550" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/cnnoutput.png">
</p>

#### Result

<p align="center">
  <img width="700" height="250" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/result.png">
</p>

### LSTM ( Long Short Term Memory )

Long Short-Term Memory, LSTM Neural Networks , is a type of recurrent neural networks that got attention recently within the machine learning community.
LSTM networks have some internal contextual state cells that act as long-term or short-term memory cells.
The output of the LSTM network is modulated by the state of these cells. This is an important property when we need the prediction of the neural network to depend on the historical context of inputs, rather than only on the very last input.

LSTM has applications in various fields such as text generation , handwriting recognition , handwriting generation , music generation , language transalation ,
image captioning , etc.

#### Output
<p align="center">
  <img width="700" height="250" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/lstmmatrix.png">
</p>

#### Accuracy and other results 
<p align="center">
  <img width="750" height="80" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/lstmaccuracy.png">
</p>

### Ensemble Model
A single algorithm may not make the perfect prediction for a given dataset. Machine learning algorithms have their limitations and producing a model with high accuracy is challenging. If we build and combine multiple models, the overall accuracy could get boosted. The combination can be implemented by aggregating the output from each model with two objectives: reducing the model error and maintaining its generalization.

Ensemble models is a machine learning approach to combine multiple other models in the prediction process. Those models are referred to as base estimators.Some textbooks refer to such architecture as meta-algorithms.
#### Ensembling Technique 
There are mainly 3 major ensembling techniques 
Bagging - Fitting decision trees on different samples of the same dataset and averaging the prediction
Stacking - Fitting many different model type on the same data and using another model to learn how to best combine their predictions 
Boosting - Involves adding ensemble members sequentially that correct the prediction made by prior models and output a weighted average of their predictions

<p align="center">
  <img width="600" height="150" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/info.png">
</p>

#### Soft Voting 
In soft voting, every individual classifier provides a probability value that a specific data point belongs to a particular target class. The predictions are weighted by the classifier's importance and summed up. Then the target label with the greatest sum of weighted probabilities wins the vote.The soft voting is often recommended in the case of an ensemble of fitted classifiers.

#### Accuracy
<p align="center">
  <img width="500" height="30" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/ensembleaccuracy.png">
</p>

#### Confusion Matrix 
<p align="center">
  <img width="300" height="100" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/ensemblematrix.png">
</p>

#### Result
<p align="center">
  <img width="750" height="80" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/ensembleresult.png">
</p>

### Comparing Results

<p align="center">
  <img width="600" height="150" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/accuracycompare.png">
</p>

### Questionnaire / FAQ 

#### What is soft voting ?
The soft voting, predicts the class label based on the sums of the predicted probabilities of the individual estimators that make up the ensemble. The soft voting is often recommended in the case of an ensemble of fitted classifiers.
#### What is hard voting ?
The hard voting type is applied to predicted class labels for majority rule voting. This uses the idea of vote for majority i.e. decisions is made in favor of whoever has more than half of the vote.
#### What are the types of voting classifier available ? 
Hard and Soft voiting are the two major type of voting classifier available.
#### What is GloVe?
GloVe stands for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus.
#### How many layers are there in an typical CNN ? 
A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer.
#### What is the difference between the convolutional approach and the simple approach in CNN ? 
Both are the same variations of cnn but number of hidden layers are different. In simple CNN generally there are two or three layers but deep CNN will have multiple hidden layers usually more than 5 , which is used to extract more features and increase the accuracy of the prediction. There are two kinds of deep CNN ,one is increasing the number of hidden layers or by increasing the number of nodes in the hidden layer. 
#### What is epoch value ? 
Training the neural network with the training data for one cycle. For epoch we use all the data at once. A forward and backward pass together are counted as one pass. An epoch is made up of one or more batches, where we use a part of the dataset to train the neural network.
#### What libraries are used in the project ? 
Some libraries used are TensorFlow , Keras , Sklearn , Glove-py , Seaborn , Numpy , Pandas , Matplotlib and more.
#### What is a Confusion Matrix ? 
A confusion matrix is a table that is used to describe the performance of a classification model (or "classifier") on sets of test data for which the true values are known.
#### What does convolutional mean in CNN?
The term convolution refers to the mathematical combination of two functions to produce a third function. It merges two sets of information. The convolution is performed on the input data with the use of a filter or kernel to then produce a map.
#### What is Pooling ? 
A layer in CNN which is one of the main building block. It functions as to progressively reduce the spatial size of the representation as to reduce the amount of parameters and computation in neural network.
#### What does convolution mean ?
A convolution is the simple application of a filter to an input that results in an activation.
#### What is RNN and how is it related to LSTM ? 
The units of an LSTM are used as building units for the layers of RNN which is often called an LSTM network. LSTM enables RNN to remember inputs over a long period of time.It is because LSTM units include 'Memory Cell' which can contain information in memory for a long period of time.
LSTM is a type of RNN.
#### Is deep learning Supervised or Unsupervised learning ? 
Deep Learning is supervised learning algorithm and it can also be applied to unsupervised learning too. 

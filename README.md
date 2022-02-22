# Infocheck 

# Fake News Detection with Deep Learning


### ||   INFOCHECK   ||     

### Fake News Detection with Deep Learning 

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



### CNN ( Convolutional Neural Networks )

For Simple CNN : 
#### Accuracy and Loss 
<p align="center">
  <img width="400" height="550" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/simplecnn.png">
</p>

For Convolutional approach : 
#### Accuracy and Loss
<p align="center">
  <img width="400" height="550" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/convocnn.png">
</p>

#### Output
Output of both the models 
<p align="center">
  <img width="400" height="550" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/cnnoutput.png">
</p>

#### Result

<p align="center">
  <img width="700" height="250" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/result.png">
</p>

### LSTM ( Long Short Term Memory )

#### Output
<p align="center">
  <img width="700" height="250" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/lstmmatrix.png">
</p>

#### Accuracy and other results 
<p align="center">
  <img width="700" height="250" src="https://github.com/vikashV3/Infocheck---Fake-News-Detection-with-Deep-Learning-/blob/main/lstmoutput.png">
</p>

### Ensemble Model and Ensemble Technique

### Questionnaire / FAQ 

#### What is GloVe?
GloVe stands for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus.
#### How many layers are there in an typical CNN ? 
A CNN typically has three layers: a convolutional layer, a pooling layer, and a fully connected layer.
#### What is the difference between the convolutional approach and the simple approach in CNN ? 
#### What is epoch value ? 
#### What libraries are used in the project ? 
#### What is confusion matrix , How do you make it ? 

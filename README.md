# Zavrsni-projekt
## Convolutional Sentiment Analysis

In this project we are gonna preprocess datasets from PyTorch and build a binary classification model for sentiment analysis.

STEP 1, import librarys, create vocabulary and iterators:
 - with pip install torchtext version 0.9.0, needed for torchtext.legacy librarys
 - import all needed librarys from pytorch
 - define TEXT and LABEL variables with corresponding parameters
 - download IMDB dataset
 - build the vocabulary with 25000 the n most common words
 - define the batch size, total number of training samples present in a min-batch
 - check if the device has a GPU
 - create the iterators
 
STEP 2, build the CNN1d model:
 - build a convolutional 1 dimensional model with the forward function
 - Init function creates word embeddings from the input words, specifys the convolution with different filter sizes and adds a fully 
   connected layer for final predictions
 - Forward function iterates through the list applying each convolutional layer to get a list of convolutional outputs
 - define parameters for the CNN1d class (enmbedding, input, output dimension, dropout)

STEP 3, train the model:
 - define optimizators (Adam optimizator)
 - define criterion with BCEWithLogitsLoss, calculates the loss
 - create two helper functions (accuracy and epoch_time) that will help to calculate the accuracy (with the sigmoid function we define the range between 0 and 1 and      mapping the positive reviews to 0.5 to 1, and the rest to negative reviews) and the time for traning the epoch
 - create functions train and evaluate, train function iteraters through all examples one batch at a time, and the evaluate function, similar to train, with some          modification, we don't want to update our parameters while we evaluate
 - define N_EPOCH, how many times we want our epoch to go through all the examples (train/evaluate)
 - evaluate the model on the saved test data

STEP 4, predict sentiment from user input:
 - build a prediction function
 - prediction function puts the model in evaluation mode, tokenizes the sentences, indexes the tokens (converts them into integers), converts the output prediction in    integer between 0 and 1 (sigmoid function), converts the tensor into integer with the item() method
 - input user reviews and check the output, positive reviews are close to 1, and negative close to 0

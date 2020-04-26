


#  **Hate Speech and Offensive Language Detection**


In this tasks I explored different machine learning techniques for classifying speech 
into 3 classes
~~~~
0 - hate speech 
1 - offensive language 
2 - neither 
~~~~
#### Created models using different algorithms
~~~~
Logistic Regressions
KDE based Bayes classifer
kernelized SVM
k-NN classier
Gaussian based Bayes classier
~~~~

The repo contains python script and ipynb file for each algorithm with
their classification report like below

~~~~

              precision    recall  f1-score   support

           0       0.25      0.20      0.22       164
           1       0.88      0.88      0.88      1905
           2       0.62      0.67      0.64       410

    accuracy                           0.80      2479
   macro avg       0.58      0.58      0.58      2479
weighted avg       0.80      0.80      0.80      2479

~~~~


### DataSet

~~~~
https://github.com/t-davidson/hate-speech-and-offensive-language/
blob/master/data/labeled_data.csv
~~~~

### Feature Design:
~~~~
#### 1. Pre-processing steps - preprocessing.py

• Removal of out of dictionary words such as tags and urls
• Tokenization
• Stop words removal
• Generation of n-grams
• Removing <2% frequency of words.
• Lemmatization (resulted in decrease of accuracy), hence removed it from preprocessing
step.

##### 2. Feature space

TD-IDF of vector of size (24783, 7086) -original feature size after preprocessing

Truncated SVD (24783, 100) – decided 100 feature size by generating different truncated SVD and 
doing thread off evaluation of feature size and accuracy for couple of algorithms.

~~~~

### Training and test datasets

Test size = 10%
Kfold split = 5
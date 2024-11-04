#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

# Create the SVM classifier
clf = SVC(kernel='linear')  # or try 'rbf' kernel for improved performance

# Train the SVM on the training set
t0 = time()  # start timing
clf.fit(features_train, labels_train)
print(f"Training time: {round(time() - t0, 3)} seconds")

# Make predictions on the test set
t1 = time()  # start timing
pred = clf.predict(features_test)
print(f"Prediction time: {round(time() - t1, 3)} seconds")

# Calculate and print the accuracy of the model
accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {accuracy}")

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# Uncomment these lines to use only 1% of the training data for faster training
# features

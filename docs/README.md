# Programming Assignment 1

Sai Santosh Kumar Ganti

02/15/2019

## Problem Definition

In this assignment, you need to build a neural network by yourself and implement the backpropagation algorithm to train it.

## Learning Objectives

1. Understand how neural networks work 
2. Implement a simple neural network 
3. Apply backpropagation algorithm to train a neural network 
4. Understand the role of different parameters of a neural network, such as epoch and learning rate.

## You need to do

1. We provide code that you may use to implement a neural network with one hidden layer. (10pts)
2. Train your neural network with the provided linear and non-linear dataset (DATA/data_linear, DATA/data_nonLinear) respectively and evaluate your trained model by a **5-fold round robin cross-validation**, i.e. separate the whole dataset into 5 parts, pick one of them as your test set, and the rest as your training set. Repeat this procedure 5 times, but each time with a different test set. To evaluate your learning system, you will need to calculate a confusion matrix. The cross validation procedure enables you to combine the results of five experiments. Why is this useful? (15pts)
3. What effect does the learning rate have on how your neural network is trained? Illustrate your answer by training your model using different learning rates. Use a script to generate output statistics and visualize them. (5pts)
4. What is overfitting and why does it occur in practice? Name and briefly explain 3 ways to reduce overfitting. (5pts)
5. One common technique used to reduce overfitting is L2 regularization. How does L2 regularization prevent overfitting? Implement L2 regularization. How differently does your model perform before and after implementing L2 regularization?(5pts)
6. **Optional for CS440, Required for CS640:** Now, let's try to solve real world problem. You are given hand-written digits as below, all digits are stored in a csv file. You need to implement the neural network class with 1 hidden layer to recognize the hand-written digits, you should train your model on DATA/Digit_X_train.csv and DATA/Digit_y_train.csv, then test your model on DATA/Digit_X_test.csv and DATA/Digit_y_test.csv. Provide your results and a discussion of the performance of your AI system. (10pts)
7. Instructions:
   1. For **CS440**, total points: **40 + 10 extra credits**. For **CS640**, total points: **50**
   2. Your report should include the results and your analysis for part 2-6. In lab section (demo), we will ask you to run and explain your code.

## Method and Implementation

We are using sigmoid function for the hidden layer and a softmax function for the output layer. 

## Experiments

Describe your experiments, including the number of tests that you performed, and the relevant parameter values.

Define your evaluation metrics, e.g., detection rates, accuracy, running time.

## Results

List your experimental results. Provide examples of input images and output images. If relevant, you may provide images showing any intermediate steps. If your work involves videos, do not submit the videos but only links to them.

## Discussion

Method and Results:

Conclusions Based on your discussion, what are your conclusions? What is your main message?

The accuracy of this model is 94% and this accuracy rate common among neural networks with one hidden layer. 

## Credits and Bibliography

Joint work and discussion with Diptanshu Sign, FNU Mohit, Abhishek Rai Sharma and Vijay Karigowdara

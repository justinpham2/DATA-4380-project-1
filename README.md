![](UTA-DataScience-Logo.png)

# Fashion Mnist Keras model

* **One Sentence Summary** :This repository holds an attempt to use KERAs to several fashion based images using data from the kaggle dataset Fashion MNIST.

## Overview

 *Definition of the tasks / challenge  : Use KERAs to create a CNN to several fashion based images using data from the kaggle dataset Fashion MNIST.
  *Your approach : Create a sequential model using several layers with the training dataset as input
  *Summary of the performance achieved** : The highest accuracy achieved is 92% with a .25 in test loss
  
## Summary of Work Done
I preprocessed the data, splitting the training set into a validation set and rescaling the images to a grayscale. Next I created a sequential model and added the layers: Conv2D, Maxpooling2D, Dropout, Flatten, and dense. Lastly I evaluated the performance of the cnn model and optimized it.  

### Data

* Type: Table of features, 28x28 images
  * Size: How much data? : 210.17mb 
  * Instances (Train, Test, Validation Split): how many data points? : 60,000 instances training set, 10,000 instances testing set, 12,000 instances validation set

#### Preprocessing / Clean up

* split data into x (image) and y (label) arrays 
* rescaled the image from 255 pixels to 0 and 1 for better convergence 
* split training set to validation set (20% of training set)

#### Data Visualization

![image](https://user-images.githubusercontent.com/98443119/226213063-a7057dbb-7eb2-49c8-aba1-b7276ba91a24.png)
![image](https://user-images.githubusercontent.com/98443119/226213074-3c35d627-5255-484f-985a-bf689656031a.png)

### Problem Formulation

* Define:
  * Input / Output:  x_train, y_train  / Output: x_validate, y_validate
  * Models: Sequential .
  * Loss, Optimizer, other Hyperparameters.:Loss= sparse_categorical_crossentropy, Optimizer = Adam, Metrics = Accuracy 
### Training

* Describe the training:
  * How you trained: Software: Tensorflow and Keras. Hardware: intel integrated GPU
  * Training curves (loss vs epoch for test/train).: The loss and epoch were consistent with each other
  * How did you decide to stop training.: When I reached high accuracy and a consistent loss/epoch rate
  * Any difficulties? How did you resolve them? When I first trained the model I got 88% accuracy with an underfitting loss/epochs. To resolve this problem I regularized the model, testing different dropout rates and adding/removing epochs.  

### Performance Comparison

* Clearly define the key performance metric(s) : Accuracy
![image](https://user-images.githubusercontent.com/98443119/226213834-a74400e6-8c9e-427a-82bc-3f8824d52128.png)

### Conclusions

* 

### Future Work

Try different layers/metrics

## How to reproduce results

*

### Overview of files in repository

* Fashion_MNIST.ipynb : the cnn model and preprocessing

### Software Setup
*!pip install tensorflow opencv-python 
*!pip install keras

### Data

https://www.kaggle.com/datasets/zalando-research/fashionmnist

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* 








![](UTA-DataScience-Logo.png)

# Fashion Mnist Keras model

* **One Sentence Summary** :This repository holds an attempt to use KERAs to several fashion based images using data from the kaggle dataset Fashion MNIST.

## Overview

 *Definition of the tasks / challenge  : Use KERAs to create a CNN to several fashion based images using data from the kaggle dataset Fashion MNIST.
  *Your approach : Create a sequential model using several layers with the training dataset as input
  *Summary of the performance achieved** : The highest accuracy achieved is 92% with a 0.25 in test loss
  
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

* In conclusion I reached a loss of 0.24 and accuracy of 0.91. 

### Future Work

Try different layers/metrics/parameters
Use tensorflow and Keras to create a base sequential model and then gather learning from an architecture and apply it to the sequential model to increase accuracy. 
Evaluate your model each time you train it in order to identify what parameters to change. 

## How to reproduce results

* To apply this package to other data:
Preprocessing: 
 Split the data into three arrays, train, test, and validation if applicable. 
 Normalize the data to convulate the model easier in creating the model 
Building the model:
 Use .CONV2D to add biases to the image. Next use .maxpooling to retain the most prominent features of an image.For models with more layers, use Batch Normalizaiton to make the neural network more stable and faster. To expand your model repeat CONV2D + maxpooling + batch normalization in that order for as many layers needed. After the last CONV2D + maxpooling + batch normalization layer, use .flatten to convert the given array to one dimenson. Next use dropout() to prevent overfitting. Use .dense at the end of your functions to classify the images. Lastly compile the model using .compile with the parameters loss, optimizer, and metrics. The loss parameter is used for multi-class classification problems, where the target variable is represented as integers. Optimizer specifies the optimization algorithm to be used during training. Lastly metrics specifies the evaluation metric to be used during training and testing.
 Notes:
  CONV2D: There are 4 parameters needed to run: filters, kernals, activation and input_shape. 
  maxpooling: There is 1 parameter needed to run: pool_size.
  Dropout: Use a value.
  Dense: There is 1 parameter: activation. 
 Training the model: 
  Use a method in keras, for example: model.kit, and 6 parameters: x_train,y_train, batch_size, epochs, verbose, and validation data. 
  
### Overview of files in repository

* Fashion_MNIST.ipynb : the cnn model and preprocessing

### Software Setup
*!pip install tensorflow opencv-python 
*!pip install keras

### Data

https://www.kaggle.com/datasets/zalando-research/fashionmnist

### Training

* This model has 6 layers: Conv2D,Maxpool2D,Dropout,Flatten,Dense,Dense. Conv2d is used to extract features and patterns. Maxpool2D reduces the spatial dimensions of the output, this is important to produce a feature map. Dropout is used to drop random units to prevent overfitting. Next is to flatten the previous layer into a 1D array. Dense adds the flatten layer to neural network. 

#### Performance Evaluation

* To evaluate the performance, I created a loss plot to read the accuracy and evaluated the loss/accuracy using tensorflow. 

### Citations
https://keras.io/api/models/sequential/









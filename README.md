# Building K-Nearest Neighbour algorithm from scratch

## Find a link to my blog post [here.](https://medium.com/where-quant-meets-data-science/building-k-nearest-neighbour-algorithm-from-scratch-bd0c5df13192)
## Objective
In this project we will be solving KNN Regression problem from scratch. We will be implementing the KNN problem in the naive method using a for loop and also in a vectorised approach using numpy broadcasting. We will also plot the root mean squared error for various K values and chose the optimal number of nearest neighbours.

## What is the K-Nearest Neighbour algorithm ?
KNN is a non-parametric supervised machine learning model which stores all the data available and predicts new cases based on a chosen similarity metric. The idea to predict the value of the new case based on the K nearest values available w.r.t to the similarity metric. It can be used for classification and regression problems.


## What are parametric and non parametric methods ?
Any machine learning model can be explained by the learning function it uses. This function maps from the X (independent) variables to the Y(dependent) variables. In a parametric model, we assume a defined learning function and try to build a model by fitting the training data to the learning function to find the model coefficients. Increase in the data would not alter the learning function of this model. Examples of parametric methods are logistic regression and simple neural networks.
Where as, in a non parametric model, we do not presume any fixed form of learning function and are free to learn any mapping function based on the training data. The most common examples are KNN and SVM.

## What is the similarity metric for KNN ?
The common similarity metric used is Euclidian distance. Alternatively we can also use manhattan distance depending on the problem. The KNN model stores all the training data set and when a new test data point is given the aim is to find K points from the training set which are closest to the test data point. So here we are trying to find the points which minimise the similarity metric.
In some high dimensional datasets, cosine similarity metric is also used where we try to find the K data points which maximise the cosine value.

## Implementing K-nearest neighbours algorithm from scratch
### Step 1: Load Dataset
We are considering the California housing dataset for our analysis. I am downloading this dataset from sklearn.I am considering a train test split of 70%:30%. Below is the code to split the data into train and test data sets.

### Step 2 : Feature Scaling
Feature scaling is an essential step in algorithms like KNN because here we are dealing with metrics like euclidian distance which are dependent on the scale of the dataset. So to build a robust model, we need to standardise the dataset. (i.e make the mean = 0 and variance = 1)

### Step 3: Naive Implementation of KNN algorithm
Here we are assuming the K values to be 10. Then we are looping over each point of the test data set, to find the euclidian distance between the test point and train data points. Then we are sorting the distance and finding the nearest 10 neighbours. Once we have the y values of the nearest neighbours, we are taking the average of those values and adjusting the prediction the appropriate mean and SD to get the predicted target variable. The RMSE turns out to be 0.82

#### Issues with this Implementation:
This naive approach would work for a small dataset but might face issues with larger data sets as we are looping through each value of the test data. This can be improved by using a vectorised approach.

### Step 4 : Vectorised implementation of KNN, using numpy broadcasting
Here we use the vectorized implementation of KNN to get the same solution as the naive implementation. 

### Step 5 : Selecting the right K value
![](/Images/RMSE.png)

We find the optimal K by extracting the K value for which the RMSE is the lowest. Turns out the optimal RMSE is around 0.82 when K is around 10–12.

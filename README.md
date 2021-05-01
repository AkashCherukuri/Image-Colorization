# Image-Colorization

This is a repo for SoC2021, Image Colorization project. This repo would contain the code and I will be using this README.md as a way to catalogue my journey throughout this project. 

### TODO

- Add LaTeX compatibility for this .md file

## Linear Regression

The main aim is to estimate a linear equation representing the given set of data. There are two approaches to this.   

1. A closed form solution.  
   This can be directly obtained by solving the linear differential equation. Calculate the partial derivative of the error function wrt x, y and equate both of them to zero to get the values of the parameters which minimizes the error function.
2. An iterative approach.  
   This is similar to **Gradient Descent**. We try to obtain the minima (L1, L2 norm etc) by calculating the gradient at each point and moving in small steps along the gradient vector. 
   Refer to [this](https://youtu.be/8PJ24SrQqy8) video for more details. 

## Logistic Regression

Refer to the following [link](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) to see an example of logistic regression. 

## Gradient Descent

[Here](https://youtu.be/sDv4f4s2SB8) is a useful video.  
An article about Gradient Descent [here](https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd)  
A useful post on GeeksForGeeks [here](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/)  

# Deep Learning

[This book](http://neuralnetworksanddeeplearning.com/index.html) on deep learning has been followed, this section will contain nomenclature and a few key definitions I thought were important enough.

- `perceptrons`: older way of representing neural networks. Can give an output of either 0 or 1, corresponding to if the value of the sum of the dot product of `w` and `x` and `bias` is lesser than or greater than 0.
- `sigmoid neurons`: we usually change the values slightly in perceptrons to slowly approach the required classification function. However, because perceptrons are binary in nature, small changes can cause drastic (and unintended) changes to the output. Sigmoid neurons try to minimize this issue.

The standard sigmoid function is given as follows:
$$ \sigma(w\cdot x+b) = \frac{1}{1+exp(-w\cdot x-b)} $$

That is, is is a smoothened out version of the step function. We can also see that the output changes linearly with changes in inputs (using partial derivatives).



## MLP - Multi Layer Perceptrons 

These have **sigmoid neurons** as layers in them. The neurons taking input are called *input neurons* and comprise the *input layer*. Similarly, we have the output neurons and the output layer. Neurons (and layers) which are neither input nor output are called as **Hidden Layers**. We will be using a **feed-forward** neural network, meaning that the output of a layer always leads to an input of another layer in a linear fashion without loops. If layers have loops, they are called **Recurrent Neural networks** or RNNs.

For example, a neural network responsible for detecting the number in a MNIST dataset can have just three layers; Input (28*28 neurons), hidden (variable `n`) and output (10 neurons). The network is trained using a training set, and the mean squared loss function is minimized by using gradient descent.

### Gradient Descent

Given a function `f(x1, x2)`, the minima of the function can be computed empirically by taking its partial derivative and “walking” such that the function value is reduced.

`\eta` is called as the *Learning Rate*, and is directly proportional to how “large” the “steps” are.

In our case, we would be applying gradient descent and changing the values of all the biases (`b_i`) and weights (`w_i`) to minimize the cost function. A drawback of this method is that calculating the cost function requires the summation of the mean squared error over all values of training data, which would be ranging in the order of `10^5`. This causes the training to be very slow.

### Stochastic Gradient Descent

Instead of taking all the `n` values in the training data set, we create a subset called the “mini set” where each element is a random subset of size `m < n`. We compute the cost function over every subset in the mini set, with the assumption that the “true” cost function and the empirically calculated cost function are nearly equal. This dramatically reduces the time required for training the network.

When the mini set is exhausted, an **epoch** of training is said to be completed after which the process is repeated again. This is to mark the progress of training.



## Resources used

A Machine Learning course [here](https://www.coursera.org/learn/machine-learning)  
Notes on Machine Learning [here](http://cs229.stanford.edu/summer2019/cs229-notes1.pdf)

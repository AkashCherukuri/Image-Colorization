# Image-Colorization

This is a repo for SoC2021, Image Colorization project. This repo would contain the code and I will be using this README.md as a way to catalogue my journey throughout this project. 


## Table of Contents

1. [Linear and Logistic Regression](#linear-regression)
2. [Deep Learning](#deep-learning)
	- [Gradient Descent and Stochastic Gradient Descent](#gradient-descent)
	- [Back Propagation](#back-propagation)
	- [Cross Entropy Cost Function](#learning-slowdown-and-the-cross-entropy-cost-function)
3. [Convolutional Neural Networks](#cnns)
	- [Basic Terminologies](#basic-terminologies)

-1. [Resources Used](#resources-used)

---


## Linear Regression

The main aim is to estimate a linear equation representing the given set of data. There are two approaches to this.   

1. A closed form solution.  
   This can be directly obtained by solving the linear differential equation. Calculate the partial derivative of the error function wrt x, y and equate both of them to zero to get the values of the parameters which minimizes the error function.
2. An iterative approach.  
   This is similar to **Gradient Descent**. We try to obtain the minima (L1, L2 norm etc) by calculating the gradient at each point and moving in small steps along the gradient vector. 
   Refer to [this](https://youtu.be/8PJ24SrQqy8) video for more details. 

### Logistic Regression

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

σ(w·x+b) = 1/(1+exp(-w·x-b))

That is, is is a smoothened out version of the step function. We can also see that the output changes linearly with changes in inputs (using partial derivatives). (w·x+b) is called as the "Weighted input" for that particular neuron, and is represented by `z`.



## MLP - Multi Layer Perceptrons 

These have **sigmoid neurons** as layers in them. The neurons taking input are called *input neurons* and comprise the *input layer*. Similarly, we have the output neurons and the output layer. Neurons (and layers) which are neither input nor output are called as **Hidden Layers**. We will be using a **feed-forward** neural network, meaning that the output of a layer always leads to an input of another layer in a linear fashion without loops. If layers have loops, they are called **Recurrent Neural networks** or RNNs.

For example, a neural network responsible for detecting the number in a MNIST dataset can have just three layers; Input (28×28 neurons), hidden (variable `n`) and output (10 neurons). The network is trained using a training set, and the mean squared loss function is minimized by using gradient descent.

### Gradient Descent

Given a function `f(x1, x2)`, the minima of the function can be computed empirically by taking its partial derivative and “walking” such that the function value is reduced.

Δf = (∂f/∂x)Δx + (∂f/∂y)Δy

   = (∇f)·(ΔX)

Let ΔX = -η∇f

Δf = -η||∇f||² , which is always negative. 

η is called as the *Learning Rate*, and is directly proportional to how “large” the “steps” are.

In our case, we would be applying gradient descent and changing the values of all the biases (`b_i`) and weights (`w_i`) to minimize the cost function. A drawback of this method is that calculating the cost function requires the summation of the mean squared error over all values of training data, which would be ranging in the order of `10^5`. This causes the training to be very slow.

### Stochastic Gradient Descent

Instead of taking all the `n` values in the training data set, we create a subset called the “mini set” where each element is a random subset of size `m < n`. We compute the cost function over every subset in the mini set, with the assumption that the “true” cost function and the empirically calculated cost function are nearly equal. This dramatically reduces the time required for training the network.

When the mini set is exhausted, an **epoch** of training is said to be completed after which the process is repeated again. This is to mark the progress of training.

*** Vectorizing sigmoid function ***

## Back-Propagation

Assumption1: The cost function for a set of inputs is equal to the average of the cost function for each individual input. This assumption holds for the Least-Mean-Squared cost function.

Assumption2: The cost function should be a function of the outputs of the neural network.

Given the cost function C, and the weighted input z for a neuron, we define error for this neuron δ as follows,

δ = ∂C/∂z

That is, if δ is a large value, then a change in z can bring about a change in C. If it is zero, then it means that C is optimal wrt z.

There are four fundamental equations to back propogation, and they have been given below. δL is the δ vector for the final layer.
- δL = (∂C/∂a) ⊙ σ'(z)
- δⁿ = (wⁿ⁺¹)ᵀ(δⁿ⁺¹) ⊙ σ'(z)
- (∂C/∂b) = δ  ⇒  Delta of a neuron is equal to the derivative of the cost function wrt to its bias
- (∂C/∂wⁿⱼₖ) = aₖⁿ⁻¹ * δⱼⁿ   (Do remember that in wⱼₖ, neuron `k` is in the n-1'th layer and neuron `j` is in the n'th layer)

This is how a single iteration of training a neural network is done:
1. Input a set of training examples.
2. Feedforward the values to get the output.
3. Calculate the error at the outer-most layer.
4. Backpropogate the error till the input layer.
5. Perform gradient descent, as partial derivatives wrt all biases and weights is known.


## Learning Slowdown and the cross entropy cost function

We've been using the quadratic cost function so far. It does have a few issues, notably its derivative is very small when the value of σ(z) is close to 0 or 1. In gradient descent, the change in biases and weights is directly proportional to the derivative of the cost function, meaning it is possible for this function to learn very slowly when it is giving wrong results. We turn to the cross entropy cost function as a solution.

C = (-1/n)×Σₓ[yln(σ(z)) + (1-y)ln(1-σ(z))] 

It can be checked mathematically that the derivative of this cost function wrt `b` and `x` is independant of σ'(z), meaning no learning slowdown occurs. Moreover, the derivative is proportional to error meaning that learning occurs faster when the model is more wrong, as we would like it.

The cross entropy cost function can be defined for an entire layer as well;-

C = (-1/n)×ΣₓΣⱼ[yln(σ(zⱼᴸ)) + (1-y)ln(1-σ(zⱼᴸ))]    where zⱼᴸ is the j'th neuron in the final layer 'L' 


Do note that a sigmoid function coupled with the cross entropy cost function is quite similar in terms of learning slowdown to the softmax function coupled with the log-likelihood cost function. (the derivatives wrt `b` and `x` have the same behaviour)


## Avoiding overfitting

1. Increase the size of training data

2. Regularization 
	- In L2 regularization, a new term is added to the cost function as shown below. The second summation is over all the weights in the network. λ is called the *regularization parameter*.

	C = (-1/n)×ΣₓΣⱼ[yln(σ(zⱼᴸ)) + (1-y)ln(1-σ(zⱼᴸ))] + (λ/2n)Σ[w²]  


	- Similarly, L1 regularization is given below:

	C = (-1/n)×ΣₓΣⱼ[yln(σ(zⱼᴸ)) + (1-y)ln(1-σ(zⱼᴸ))] + (λ/n)Σ|w|


	- Dropout regularization is a technique wherina random half of the hidden neurons are ommited from the network for a single training iteration. The idea here is that "different networks can have different overfitting heuristics, and training them seperately can cause the averaging out of their errors."

3. Artificially inflate the size of training data
	In the case of MNIST, just rotate/blur the images by a small degree to get new training data!

## Initializing the weights and biases

We have so far been initializing all weights and biases from a gaussian distribution of mean 0 and standard deviation 1. This isn't optimal, as the standard deviation of `z = (Σᵢwᵢxᵢ) + b` would be very large, proportional to the square of the umber of inputs the neuron has. This can cause the output of the sigmoid function to be nearly 0 or 1, causing stagnation as discussed earlier.

To solve this problem, we initialize `b` as earlier but `w` is initialized with mean 0 and standard deviation of `1/sqrt(n)` where n is the number of inputs.


## Universality of Neural Networks

This is a very important mathematical analysis (which I shall not write here for the sake of brevity) that neural networks (even simple ones with a single hidden layer) can compute any function with relative precision given enough time to train.

The approximation is better when there are more neurons used in the hidden layer. Also, we get an approximated continuous function as a result of estimating a discontinuous function by this method.

---


# CNNs

Stand for Convolutional Neural Networks. Are best suited for image processing, and the number of connections in between the hidden layers is decreased by a significant factor. A few basic terminologies are discussed below.

## Basic Terminologies
1. Convolution

	This step involves having a *kernel* of fixed size move across the image (with a stride that need not be 1) and produce a *feature map* which makes it easier for the network to train. Many kernels can be operated on a single image, giving many feature maps.

	Do note that the kernel must always be odd-sized.

2. Pooling

	The size of a map is reduced by first dividing it into smaller parts, each with size m×m. Each of these smaller squares is replaced by a single pixel, usually by taking the `max` or the `average` of all the values in that cell.

3. ReLU

	This introduces non-linearity in the system so as to make the network more flexible in representing a large variety of functions.

4. Fully Connected Layers

	Once all the "pre-processing" of the image via convolution and pooling is done, the resultant values are passed into a fully connected layer for the neurons in that layer to train. The number of layers is variable, but the output layer must have the sam enumber of neurons as the number of possible classifications. (due to obvious reasons)   


### Types of Convolution
1. Dilated Convolution

	Converting a 10x10 map to a smaller map of size 6x6 using a kernel of size 3 would take two consecutive convolutions. Instead of doing this twice, we can "inflate" the size of our original kernel to 5 by adding two additional rows and columns of 0s in between. This would require the convolution to be done only once, saving computational effort.

	The number of rows/columns of 0s added is called as *Dilation Rate*.

	Example:

						1 0 1 0 1

	1 1 1 			0 0 0 0 0 

	1 1 1 	⇒ 		1 0 1 0 1

	1 1 1 			0 0 0 0 0 

						1 0 1 0 1

2. Transposed Convolution

	This is the reverse of convolution, where we increase the size of the map by padding it and applying the feature map. This is used to "estimate" what the original map might've been, and is used in encoder-decoder networks.


## Resources used

A Machine Learning course [here](https://www.coursera.org/learn/machine-learning)  
Notes on Machine Learning [here](http://cs229.stanford.edu/summer2019/cs229-notes1.pdf)

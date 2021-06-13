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
	- [Types of Convolution](#types-of-convolution)
	- [Famous CNNs](#famous-cnns)

4. [Generative Neural Networks](#gans)
	- [Inverse Transform Method](#inverse-transform-method-and-its-implications)
	- [Generative Models](#generative-models)
	
-1. [Resources Used](#resources-used)

---
&nbsp; 

&nbsp; 

&nbsp; 


## Linear Regression

The main aim is to estimate a linear equation representing the given set of data. There are two approaches to this.   

1. A closed form solution.  
   This can be directly obtained by solving the linear differential equation. Calculate the partial derivative of the error function wrt x, y and equate both of them to zero to get the values of the parameters which minimizes the error function.
2. An iterative approach.  
   This is similar to **Gradient Descent**. We try to obtain the minima (L1, L2 norm etc) by calculating the gradient at each point and moving in small steps along the gradient vector. 
   Refer to [this](https://youtu.be/8PJ24SrQqy8) video for more details. 

### Logistic Regression

Refer to the following [link](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) to see an example of logistic regression. 

### Gradient Descent

[Here](https://youtu.be/sDv4f4s2SB8) is a useful video.  
An article about Gradient Descent [here](https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd)  
A useful post on GeeksForGeeks [here](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/)  

&nbsp; 

&nbsp; 

&nbsp; 

# Deep Learning

[This book](http://neuralnetworksanddeeplearning.com/index.html) on deep learning has been followed, this section will contain nomenclature and a few key definitions I thought were important enough.

- `perceptrons`: older way of representing neural networks. Can give an output of either 0 or 1, corresponding to if the value of the sum of the dot product of `w` and `x` and `bias` is lesser than or greater than 0.
- `sigmoid neurons`: we usually change the values slightly in perceptrons to slowly approach the required classification function. However, because perceptrons are binary in nature, small changes can cause drastic (and unintended) changes to the output. Sigmoid neurons try to minimize this issue.

The standard sigmoid function is given as follows:

σ(w·x+b) = 1/(1+exp(-w·x-b))

That is, is is a smoothened out version of the step function. We can also see that the output changes linearly with changes in inputs (using partial derivatives). (w·x+b) is called as the "Weighted input" for that particular neuron, and is represented by `z`.

&nbsp; 

## MLP - Multi Layer Perceptrons 

These have **sigmoid neurons** as layers in them. The neurons taking input are called *input neurons* and comprise the *input layer*. Similarly, we have the output neurons and the output layer. Neurons (and layers) which are neither input nor output are called as **Hidden Layers**. We will be using a **feed-forward** neural network, meaning that the output of a layer always leads to an input of another layer in a linear fashion without loops. If layers have loops, they are called **Recurrent Neural networks** or RNNs.

For example, a neural network responsible for detecting the number in a MNIST dataset can have just three layers; Input (28×28 neurons), hidden (variable `n`) and output (10 neurons). The network is trained using a training set, and the mean squared loss function is minimized by using gradient descent.

&nbsp; 

### Gradient Descent

Given a function `f(x1, x2)`, the minima of the function can be computed empirically by taking its partial derivative and “walking” such that the function value is reduced.

Δf = (∂f/∂x)Δx + (∂f/∂y)Δy

   = (∇f)·(ΔX)

Let ΔX = -η∇f

Δf = -η||∇f||² , which is always negative. 

η is called as the *Learning Rate*, and is directly proportional to how “large” the “steps” are.

In our case, we would be applying gradient descent and changing the values of all the biases (`b_i`) and weights (`w_i`) to minimize the cost function. A drawback of this method is that calculating the cost function requires the summation of the mean squared error over all values of training data, which would be ranging in the order of `10^5`. This causes the training to be very slow.

&nbsp; 

### Stochastic Gradient Descent

Instead of taking all the `n` values in the training data set, we create a subset called the “mini set” where each element is a random subset of size `m < n`. We compute the cost function over every subset in the mini set, with the assumption that the “true” cost function and the empirically calculated cost function are nearly equal. This dramatically reduces the time required for training the network.

When the mini set is exhausted, an **epoch** of training is said to be completed after which the process is repeated again. This is to mark the progress of training.

*** Vectorizing sigmoid function ***

&nbsp; 

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

&nbsp; 

## Learning Slowdown and the cross entropy cost function

We've been using the quadratic cost function so far. It does have a few issues, notably its derivative is very small when the value of σ(z) is close to 0 or 1. In gradient descent, the change in biases and weights is directly proportional to the derivative of the cost function, meaning it is possible for this function to learn very slowly when it is giving wrong results. We turn to the cross entropy cost function as a solution.

C = (-1/n)×Σₓ[yln(σ(z)) + (1-y)ln(1-σ(z))] 

It can be checked mathematically that the derivative of this cost function wrt `b` and `x` is independant of σ'(z), meaning no learning slowdown occurs. Moreover, the derivative is proportional to error meaning that learning occurs faster when the model is more wrong, as we would like it.

The cross entropy cost function can be defined for an entire layer as well;-

C = (-1/n)×ΣₓΣⱼ[yln(σ(zⱼᴸ)) + (1-y)ln(1-σ(zⱼᴸ))]    where zⱼᴸ is the j'th neuron in the final layer 'L' 


Do note that a sigmoid function coupled with the cross entropy cost function is quite similar in terms of learning slowdown to the softmax function coupled with the log-likelihood cost function. (the derivatives wrt `b` and `x` have the same behaviour)

&nbsp; 

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

&nbsp; 

## Initializing the weights and biases

We have so far been initializing all weights and biases from a gaussian distribution of mean 0 and standard deviation 1. This isn't optimal, as the standard deviation of `z = (Σᵢwᵢxᵢ) + b` would be very large, proportional to the square of the umber of inputs the neuron has. This can cause the output of the sigmoid function to be nearly 0 or 1, causing stagnation as discussed earlier.

To solve this problem, we initialize `b` as earlier but `w` is initialized with mean 0 and standard deviation of `1/sqrt(n)` where n is the number of inputs.


## Universality of Neural Networks

This is a very important mathematical analysis (which I shall not write here for the sake of brevity) that neural networks (even simple ones with a single hidden layer) can compute any function with relative precision given enough time to train.

The approximation is better when there are more neurons used in the hidden layer. Also, we get an approximated continuous function as a result of estimating a discontinuous function by this method.

&nbsp; 

---

&nbsp; 

&nbsp; 

# CNNs

Stand for Convolutional Neural Networks. Are best suited for image processing, and the number of connections in between the hidden layers is decreased by a significant factor. A few basic terminologies are discussed below.

## Basic Terminologies
1. Convolution

	This step involves having a *kernel* of fixed size move across the image (with a stride that need not be 1) and produce a *feature map* which makes it easier for the network to train. Many kernels can be operated on a single image, giving many feature maps.

	Do note that the kernel must always be odd-sized.

	Check [this](https://datascience.stackexchange.com/questions/9175/how-do-subsequent-convolution-layers-work) link to see how two successive convolutions are done.

2. Pooling

	The size of a map is reduced by first dividing it into smaller parts, each with size m×m. Each of these smaller squares is replaced by a single pixel, usually by taking the `max` or the `average` of all the values in that cell.

3. ReLU

	This introduces non-linearity in the system so as to make the network more flexible in representing a large variety of functions.

4. Fully Connected Layers

	Once all the "pre-processing" of the image via convolution and pooling is done, the resultant values are passed into a fully connected layer for the neurons in that layer to train. The number of layers is variable, but the output layer must have the sam enumber of neurons as the number of possible classifications. (due to obvious reasons)   

&nbsp; 

### Types of Convolution
1. Dilated Convolution

	Converting a 10x10 map to a smaller map of size 6x6 using a kernel of size 3 would take two consecutive convolutions. Instead of doing this twice, we can "inflate" the size of our original kernel to 5 by adding two additional rows and columns of 0s in between. This would require the convolution to be done only once, saving computational effort.

	The number of rows/columns of 0s added is called as *Dilation Rate*.

	Example:
	```

	 			1 0 1 0 1 

	1 1 1 			0 0 0 0 0 

	1 1 1 	⇒ 		1 0 1 0 1

	1 1 1 			0 0 0 0 0 

	 			1 0 1 0 1

	
	```
2. Transposed Convolution

	This is the reverse of convolution, where we increase the size of the map by padding it and applying the feature map. This is used to "estimate" what the original map might've been, and is used in encoder-decoder networks.

&nbsp; 

## Famous CNNs
- **LeNet**

- **AlexNet**

	Implementation of AlexNet using `PyTorch` is given in `AlexNet.py` file. An accuracy of 95% was acheived after training the net for 15 epochs, although it seemed to saturate after the first epocg itself.

- **VGG**

	This has a very similar implementation philosophy as AlexNet, we increase the number of feature maps while decreasing the size of each feature map. A small improvement here is that convolution layers are put successively, so as to save computational time.

	That is, a 7x7 kernel over `c` sized map would need 49c² wheras having two succesive 3×3 kernels would need 27c² computations.

	This has been implemented in `VGGNet.py`.

- **GoogleNet** (2014 Winner of ImageNet challenge)

	Convolution with kernel size larger than (or equal to) 3 can be very expensive when the number of feature maps is huge. For this, GoogleNet has convolutions using a kernel size of 1 to reduce the feature maps, followed by the actual convolution. These 1×1 feature maps are also called as *Bottle Neck* feature maps. 

	GoogleNet also utilizes Inception Modules.

- **ResNet** (2015 Winner of ImageNet challenge)

	Short for residual network, the net submitted for the challenge had 152 layers but the number of parameters are comparable to AlexNet (Which has just 4 layers). The problem of learning slowdown is tackled in a very novel way in this network.

	ResNet acheives this by having "skip connections" in between the blocks of convolution. That is, let an input `X` be passed into a convolution block to get an output `F(X)`. The skip connection is used to add `X` to this result, yielding `X+F(X)`. The reasoning is that although the network might fail to learn from `F(X)`; it will be able to learn from `X` itself directly.

	My implementation of ResNet for the MNIST dataset has been shown in `ResNet.py`. I have been able to acheive an accuracy of 98%, but I do believe 99% is possible had I trained the net for longer. The feature maps and structure of the network has been modified a little to make it more easier to train, but the core idea remains the same.

&nbsp; 

---

&nbsp; 

&nbsp; 


# GANs

GAN stands for Generative Adversial Networks, and they belong to the set of Generative models. These models are called "generative" because of their ability to generate new data, such as a picture of a human face.

The mathematical analysis and intuition behind these networks is given below.

&nbsp; 

## Inverse Transform Method and its implications

This method is used to simulate drawing values from a complicated probability distribution, using a uniform distribution. Let the CDF of the complicated distribution be `F`. Assume that it is invertible, and let its inverse be given by `F⁻¹`. Let a draw from the uniform distribution be `u`. It can quite easily be proven that `F⁻¹(u)` has the same distribution as `F` itself.

`F⁻¹` is called as the **Transform Function**, as it is used to transform the uniform distribution to the target distribution.

This has a very important implication in the generation of new data. Suppose that there is a probability distribution for a vector of size n² (an image of size nxn) called the "Human Distribution" `H`, which tells how likely it is that the image represents a human face. We can now *simply* generate a random vector from uniform distribution, pass it through `H⁻¹` to generate a human face!

There are very obvious problems here:
- The "human distribution" is a very complex distribution over a very large space
- It may or may not exist

&nbsp; 

## Generative Models

In reality, we cannot explicitly say what `H` is. Therefore, a generative model aims at estimating the value of the transform function (`H⁻¹` in this case). Training such a model has two ways, **direct** and **adversial**. Both these methods have been explained below.

&nbsp; 

### Direct training (Generative Matching Networks)

We have one neural network which aims to estimate the transform function by comparing the acheived distribution to the "true" distribution. However, we do not know the true distribution (otherwise we wouldn't need to do all this!) We thus take samples of human faces from available data and generate human faces via the neural network. Then, the **M**aximum **M**ean **D**iscrepency between the two estimated distributions is taken as the error for back-propogation. 

The neural net would strive to reduce MMD, meaning that it would learn the transform function upon training for sufficient time. This way of training for generation is used in Generative Matching Networks.

&nbsp; 

### Adversial training (GANs)

We have two neural networks here, the *generator* and the *discriminator*. The generator has the same role as the previous model, wherein it tries to estimate the transform function by generating images of a human face. The discriminator is handed images from both generator and the already available data. It is tasked with classifying the images into two groups, the ones from the generator and the ones from the true data set. The generator is hence tasked with fooling the discriminator to the best possible extent.

Both the networks are trained simultaneously, and it can be clearly seen that they both are competing with each other. The discriminator tries to increase the error between the generated and the true data set whereas the generator tries to decrease this value at the same time. This competition is why the architecture is referred to as an "Adversial" network. Both the nets improve in what they're trying to acheive by competing against each other.


&nbsp; 

---

&nbsp; 

&nbsp; 

## Resources used

- [Neural Networks and Deep Learning by Miachel Nielson](http://neuralnetworksanddeeplearning.com/index.html)
- A Machine Learning course [here](https://www.coursera.org/learn/machine-learning)  
- Notes on Machine Learning [here](http://cs229.stanford.edu/summer2019/cs229-notes1.pdf)
- [Understanding Generative Adversarial Networks](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)
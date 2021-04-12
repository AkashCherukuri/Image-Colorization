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

## Resources used
A Machine Learning course [here](https://www.coursera.org/learn/machine-learning)  
Notes on Machine Learning [here](http://cs229.stanford.edu/summer2019/cs229-notes1.pdf)

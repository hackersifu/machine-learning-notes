# machine-learning-notes
Notes and artifacts about Machine Learning

Learning Pytorch and Juniper Notebooks is a must

# Overall Notes
May of these concepts haven't changed in a long time
Math is math, and this is math.
Learning core fundamentals of ML will make learning ML in the future much easier, but it does take time because it is A LOT

# Lab 1 Notes

numpy
#### numpy is the mathematical foundation for most scientific python libraries. 
#### it largely wraps Fortran to provide fast matrix operations. Fortran. 

#### scikit-learn is a library that provides many machine learning and data science primitives.
#### california Housing is an example dataset that we will use later
from sklearn.datasets import fetch_california_housing

.reshape - Learn about reshape
Z.reshape(-1) works, but seems to not care about what negative is there
Typically use Z.reshape(1,-1), this uses what is called an unknown value (-1)
np.moveaxis(<variable>,0,-1) = does a reshape on the existing variable image dimensions

axis - axis refers to a specific dimension of an array along which operations should be performed. For example, in a 2-dimensional array, the first dimension (rows) is axis 0, and the second dimension (columns) is axis 1. Operations like sum, mean, or maximum allow you to specify the axis along which the operation should be applied. A lot of NumPy functions allow you to specify an axis argument, if you don't provide this argument NumPy will assume axis=0.

How does axis know it works with 4 sets????????

Machine Learning works on NUMBERS
Feature extraction - turns inputs we want to analyze (images, words, etc) into arrays of numbers
Models can be extremely pick about the size and shape of these arrays

Basic Data Structures:
- Scalar: a single number
- Vector 1-dimensional list of numbers
- Matrix 2-Dimensional array of numbers
- Tensor - a general term for a n-dimensional array
- Array 
- Batch dimension
    - Data scientists think in batches

a python list as a 1 dimentional array
list of lists are matrices

dtype = Data types (maybe?)
uint8 = intengers
float32 = ?

Reshaping only changes the dimensions. When you see a shape error, think carefully about the mechanics going underneath.
Biggest issues are going to be batching and dtypes?

np.arange() - generate a range of numbers in an array
array.reshape (SUPER IMPORTANT) Reshape an array into a particular shape

Like Python lists, but designed for efficent computation

It might be helpful to think of these data structures and their properties like a protocol for a model.

A model is a simplified or compressed representation of a system.
1. You have some data that describes a real world $THING
2. You need a way to reason about it
3. Create a model or representation that helps you understand it.
Models do not always involve math
The rational actor model
Maslow's Hierarchy of Needs
Threat Model

1. You have a text from the internet that represent knowledge
2. You turn text into numbers, being careful to ensure they are representative
3. Use math to build (train) a model that represnts the numbers
4. Deploy it somewhere to be used
Classification models - Models that seperate data into bins
Generative models - Models that learn sequences
Physics Models - Wind tunnels, control plane coefficeients

Frame the problem
The task - What do you want the model to do? How are you going to measure success?
The data - What training data do you have? How can you turn it into a form ML model like (i.e. numbers)
The model - For any given task, there are many algorithms or architectures that may be able to solve that task.
Do you even need ML?

Linear models (and logistic regression for classification)
Very simple, very interpretable, often "good enough"
Can do well with good feature engineering
Extremely sensitive to weird data

Other models
- Decision Trees - Often not very good, fairly interpretable, rarely used in practice (not powerful enough, interpretablility is often not a requirement)
    - Not really going to talk about it in this course.

Solving is a function of user-acceptance
- All models are wrong, but some are useful.
We should expect models to be "mostly good most of the time".

Logisitic fuction (void function) - things come out 0 or 1 (bound to 0 and 1)

reshape and moveaxis are very different, need to learn some differences

### dtypes
The dtype determines how much memory each element takes and what range of values it can represent. Choosing the right dtype allows optimizing storage efficiency and computational precision, and also setting or assuming the wrong dtype will cause errors in your code. For example, take pixel values, 255.0 and 255 are the same value, they are not the same dtype.
Typical dtype is float64

int8 - 8-bit signed integer (-128 to 127): np.array([1, 2, 3], dtype=np.int8)
uint8 - 8-bit unsigned integer (0 to 255): np.array([1, 2, 3], dtype=np.uint8)
int16 - 16-bit signed integer (-32768 to 32767): np.array([1, 2, 3], dtype=np.int16)
float16 - 16-bit half-precision float: np.array([1.2, 3.45, 5.678], dtype=np.float16)
int32 - 32-bit signed integer (-2147483648 to 2147483647): np.array([1, 2, 3], dtype=np.int32)
float32 - 32-bit standard floating point: np.array([1.2, 3.45, 5.678], dtype=np.float32)
int64 - 64-bit signed integer: np.array([1, 2, 3], dtype=np.int64)
float64 - 64-bit double-precision float: np.array([1.2, 3.45, 5.678], dtype=np.float64)

Linear Regression
- Establish relationships between a dependent variable and one or more independent variables
- X typically represents the independent variables. Where big X refers to hours_studied, and little x refers to the number of hours studied for a single exam.
- Y typically represents the dependent variables. Where big Y refers to corresponding_grades, and little y refers to the grade of the corresponding exam (little x).

polyfit - note to learn about it

y = mx + b

From exercise 7:
What might be some limitations of this approach?
What application-level checks might you want to implement before passing data to the model for a prediction?
Where are you least confident in prediction results?
What you should get from this exercise:

See predictions from a simple model
You should try to get a negative value out of the model, as well as a prediction greater than 100.
You should see what happens if you put in negative hours.
This should get you started thinking about assumptions that are implicit in the model as well as seeing what happens when you violate them. This is a key part of assessing a model from a security point of view. Standard "attacking models" trick: try extreme values. <----------------------- THIS

Logistic Regression - a yes/no value based on data
A common trick to convert a model that is used for regression into one that can be used for prediction is called a response transformation - we use a linear model to produce a score, and then pass that score through a 'squashing function' that will force it's value into the range 0 to 1. We then interpret that squashed score as a probability of a particular result and set an arbitrary threshold at a point that gives us the best overall error rate (you already know this one).

The logistic function that is used to turn a number that can be anything from 
−
∞
−∞ to 
+
∞
+∞ and turns it into a number between 0 and 1. In the case of logistic regression, it's often interpreted as a probability. The logistic function is defined as so:

p
(
x
)
=
1
1
+
e
−
x
p(x)= 
1+e 
−x
 
1
​
 
If we're talking about grades, we might be looking at pass-fail scores; maybe it's curved so that half the class (those with scores above the median) passes and the rest fail.



# Lab 2 Notes - Neural Networks
Used Zillow as an example.
http://matrixmultiplication.xyz

Layers
- Input layers
- Hidden layers
- Output layers

Gradient Descent
Loss Functions
- How wrong is this answer?
- We learn a decision boundary that minimizes the average loss across all the training data.
- Large distance from the decision boundard + on the wrong side = huge lose
- Gradient Descent - we want to minimize the loss. Gradient tells how to change the parameter to do that.
- Gradient = slope of the function at that precise point
    - or: which direction is uphill?
- Negative gradient?
    - Uphill is a negative direction. Increase the parameter value to go downhill
Positive gradient
    - Uphill is in the positive direction
    - Decrease the parameter value to god downhill
Gradient = 0
    - You're done!
You can replace gradient with slope
Loss function - How poorly does this current model perform?

Pytorch does a lot of the math for us
- backward() can only be used once per model call

As long as you can traslate the data into numbers, Neural networks can be used for any problem.

```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
```

torch: The main PyTorch library.
torch.nn as nn: This module provides various classes and functions to build neural networks. It contains components like layers (e.g., Linear, Conv2d, LSTM), activation functions (e.g., ReLU, Sigmoid, Tanh), and loss functions (e.g., CrossEntropyLoss, MSELoss).
torch.optim: The torch.optim module contains optimization algorithms like Stochastic Gradient Descent (SGD), Adam, and RMSprop, which are used to update the weights of a neural network during training.

Pytorch also provides some useful libraries for working with specific types of data (torchaudio, torchtext, and of course torchvision) and pipeline components.

from torchvision import datasets: torchvision is an additional package built on top of PyTorch, specifically designed for working with computer vision tasks. The datasets submodule provides access to popular computer vision datasets like MNIST, CIFAR-10, and ImageNet.
from torchvision import transforms: The transforms submodule contains various image preprocessing functions, such as resizing, normalization, and data augmentation techniques.

Remember that torch is "NumPy for the GPU". The GPU is a component that, as you might imagine, is very efficient at matrix multiplication. It has other benefits too - it increases the white noise in your office, helping you focus on your work, and it can keep you warm in the winter months too. torch has distributed training, which allows you to specify the mechanics of which we will explore in the Large-Language Model labs.

Pytorch uses the torch.Tensor class to hold arrays and to help track the information that is required to compute gradients, as well as to move the arrays back and forth between the GPU and CPU. The requires_grad parameter is used to specify whether the gradients should be calculated for a tensor during backpropagation. When set to True, it indicates that the gradients for that tensor will be computed and accumulated during the backward pass. This is useful for trainable parameters that you want to optimize like weights and biases.

If requires_grad=False, the tensor is marked as a leaf node in the computation graph, and gradients are not calculated for it during .backward(). This is more efficient for tensors that are not being trained like fixed inputs or labels. All you need to remember is:

requires_grad=True: Compute gradients for this tensor during backprop
requires_grad=False: Do not compute gradients, treat as a fixed tensor

You can convert numpy arrays to torch tensors with the torch.from_numpy or the torch.tensor function. Alternatively, if you want to convert a torch.tensor to an numpy array, you can use X.numpy(). This mechanic is worth remembering, while GPUs can accelerate computation, moving data back and forth can be expensive (takes a lot of time). While there are accelerated data analysis libraries like RAPIDS or cuDF, most often you will load data onto the CPU to play with it.

## Basic Evasion
ML Evasion requires:
1. A model
2. A sample
3. A desired result
4. While preserving some intrinsic property of the sample
Kind of sample/ML model/Model result/Preserved Property

The goal is to keep the intrinsic property (human still sees the real value, like a bus), but the model thinks differently (an ostrich)


Relevant Links:
- http://websocketstest.courses.nvidia.com - Check on the browser and plugins for compatibility
- learn.nvidia.com - main site
- http://matrixmultiplication.xyz - for reviewing matrix multiplication

## Lab: Basic Modeling
1. Explore data structures and some reshaping
2. Introduce linear models
3. Introduce linear models + evaluations

## (Pytorch) How to install

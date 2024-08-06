# machine-learning-notes
Notes and artifacts about Machine Learning

Learning Pytorch and Juniper Notebooks is a must

# Overall Notes
May of these concepts haven't changed in a long time
Math is math, and this is math.
Learning core fundamentals of ML will make learning ML in the future much easier, but it does take time because it is A LOT
NumPy is for the CPU and Torch is for the GPU
Everything is a function, it doesn't matter how many models make up something, it just makes up a bigger function. Attackers need to care about inputs/outputs.
Typically, Security follows frameworks/steps/etc. AI Security follows no such model.
Finding eval is fun as a pentester (code execution)
AI Security is really Application Security
ML is hard! Have empathy for the process and knowledge. ML folks have worked hard.
If you don't want PII in your model, don't put it into the model!
- Just because you found it publicly, doesn't mean it should be.
Review models on HuggingFace! Are you testing against a known model?

# Notes to feed ChatGPT
Describe what shape and reshape are in machine learning.
Describe what normalize and unnormalize are in machine learning.
Gradient Descent
What is `torch.clamp`?
Describe what squeeze and unsqueeze are in machine learning.
Describe how to use `import matplotlib.pyplot as plt` properly in machine learning.
### Tool Questions
What is this: `from alibi.explainers import AnchorText`
What is this: `from alibi.utils import spacy_model`
How do I use `import zipfile` in Python?

# Need to learn
Dataset
- A torch Dataset is simply a Python class that inherits from `torch.utils.data import Dataset`, and implements __init__, __len__, and __getitem__.
.reshape - Learn about reshape
Z.reshape(-1) works, but seems to not care about what negative is there
Typically use Z.reshape(1,-1), this uses what is called an unknown value (-1)
np.moveaxis(<variable>,0,-1) = does a reshape on the existing variable image dimensions
what is a NaN? NaNs are a special value that indicate an undefined or unrepresentable value. NaN could be the result of a parsing error or be used to represent invalid operations like 0/0, ∞ - ∞. Libraries usually provide mechanisms to gracefully handle potential NaNs in code instead of outright crashing.
Learn what TfidVectorizer is `from sklearn.feature_extraction.text import TfidfVectorizer`

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

BayesianRidge - need to know what this is



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

the process for evading a model is the same across pretty much all modalities -

Content filters: the useful property might be getting phish past a classifier, while the machine learning system being evaded is a spam filter.
Facial Recognition: You wear a mask that obfuscates your face
Sentiment classifiers: It looks like a positive review to a human, but a scathing condemnation to the model.
Malware detector: The ML model labels a malicious file as benign despite it executing the same malicious functionality.

When you're attacking models, access typically comes in one of three flavors:

Gradient Access: You have complete access to the model, either by stealing it, or by the victim using a pretrained model that you have identified and found your own copy of on say, HuggingFace. In this case, you can use Whitebox gradient-based attacks. These attacks use the weights (parameters of the model).
Scores: You have API access to the model which provides complete numerical outputs of the model. In this case, you can use methods that estimate the gradients from the scores, {"class_0: "0.89, class_1: 0.06, class_2: 0.049, class_3: 0.01}. These attacks use "soft" labels, or probabilities from the output.
Labels: You have API access to the model that only returns the label of the input (or, occasionally, the top 5 or so labels). In this case, you are often forced to use noisier techniques that estimate gradients based on sampling. [class_0, class_1, class_2, class_3]. These are "hard" labels, and represent the represent the most difficult targets, with [class_0, class_1] theoretically being the most difficult. Some algorithms (like HopSkipJump) can use any access.

unnormalize (NEED TO LEARN MORE ABOUT THIS) - this function takes an input image tensor that has been preprocessed for our model and reverses the preprocessing steps to return the original image such that we can view, save, and look at it with our human eyes.

The reason is looks like it does, is because torch doesn't have an UnNormalize function. Fortunately, we can still recover our original image by reversing the normalization operation performed during preprocessing. This is done by multiplying the image tensor with the standard deviation and adding the mean values for each channel.

The unsqueeze function creates a batch dimension, which is effectively an index to a sample. For example, img_tensor[0] is simply the image we loaded as a tensor of shape (3, 224, 224). Because we train or predict on a lot of samples at once, remembering to create a batch dimension is important. But don't worry, torch will yell at you if you get it wrong.

Some attacks are built for specific modalities, for example,

Attack	Modality	Label
PixelAttack	Images	Soft
DeepWordBug	Text	Any
HopSkipJump	Any	Any
But most models, so long as you get the shapes and dtypes right will give you back a prediction.

```
# Code sample of an attack:
    output = model(img_tensor+mask_parameter)
    total_loss, class_loss, l2_loss = loss_function(output, mask_parameter, current_index)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("Total loss: {:4.4f}    class loss:{:4.4f}     l2 loss: {:4.4f}   Predicted class index:{}".format(
        total_loss.item(), class_loss.item(), l2_loss.item(), output[0].argmax()
    ))

```

The goal is to keep the intrinsic property (human still sees the real value, like a bus), but the model thinks differently (an ostrich)

Finding evasions is similiar across all inputs
1. We just need to figure out how they represent inputs as numbers
2. Define success

3 different attacks
- Whitebox - You have complete access to the model, either by stealing it or by the victim using a pretrained model that you have identified.
- Graybox - You have API access to the model which provides complete numerical outputs of the model. In this case, you can use methods that estimate the gradients from the scores.
- Blackbox - You hae API access to the model that only returns the label of the input. In this case, you're often forced to use noisier techniques.

Decision boundary
Adversarial pertuberation

### HopSkipJump
1. Find something, ANYTHING, on the target side of the decision boundary.
2. Use our starting image, and what we just found as "anchors" to find a point on the decision boundary between them.
3. Use some fancy math (what) to estimate the gradient at the decision boundary. Jump from the point on the decision boundary

HopSkipJump by Hand
The previous examples all relied on being able to compute or estimate model gradients based on probalisitic outputs to find an extremely efficient adversarial perturbations extremely quickly. In practice however, we're often attacking a model on the other side of an API and only get labels (no scores or probabilities), dog, cat, sheep, parrot, etc. In this case, there are better algorithms to choose - The most widely used one is "HopSkipJump" (if only because we tend to use it the most) -- the details of this approach are complex, but the general plan of attack is as follows:

Add random noise to get any kind of misclassification -- alternatively, use a second image.
Interpolate between the images until you find the mask that just barely changes your starting image to the new class.
Estimate the best direction to go to increase the misclassification strength; "jump" in that direction to create a new second image.
Repeat step 2 and 3 until you get a stable result.

Bisection search (what is this?)

Takeaways:
- Everything is optimizable: your model, inputs, etc
- We don't know the decision boundary
- The key weakness in most ML models is that the space of possible inputs is huge relative to the space of training and test data. There are always gaps. Exploit them.
- There are more types of evasion attacks than adversarial images, like text

# Extraction
Build a "functionally equivalent" (you define this) copy of the target model.
Extraction is where you get to be a data scientist.
- What data and features do you care about?
- What accuracy/uncertainty metrics do you care about?
- How are you going to generate/collect/augment that data?
- When are you done?

General Pattern
- Identify target
- Generate/collect training data
    - Separates the pros from the joes
    - Do you need your proxy to approximate the target across the entire feature space?
- Use a victim model to label the training data
- Use the labels from the model  to train your proxy model
    - What if we used these ad feedback for Step 2

Tradecraft
- What data are you using to generate labels?
- What auth model does the target have and what does that mean for detection?
- Ensembles, do you care?
- Data wrangling
- Model versioning and testing
- What if they target is ensembled or evolving? A/B tests?

Doing the basics
- Get good with working with data
- Understand features, how they're distributed, and how they relate (especially to your target variable)
    - Correlation is an easy way to do this.
- Your data is almost always multidimensional, but data visualization is best done in 2/3D. What can you do?
    - Having a mental model for how your data is distributed
- How can unsupervised learning help?

Unsupervised learning and dimension reduction
- Uncovers structure in data = don't need labels
- Often, just think "fancy clustering"
- Common approaches
    - K-means clustering
    - PCA (Principal component analysis)
        ```
        from sklearn.decomposition import PCA
        ```
    - T-SNE (mostly for visualization)
    - Hierarchical clustering

Extraction needs 3 things:
1. The ability to submit inputs and observe outputs (we need feedback to learn from): Attacks in the wild are difficult if only because getting direct access to inputs and outputs is not a given. In this lab setting we have all the information we could possibly need to ensure success. We get clean logits on the output, we have access to weights, we know exactly what features the model requires. In this regard, the experiment we have set up is our "best case" scenario. However, in most cases you will interact with systems that require to upload an image of bytes, not a vector. Or did you notice that we've been working in batches of 10 (torch.FloatTensor(10, 1, 28, 28))? Not all applications support batches, meaning you need to come up with your own batching scheme.

2. A representative dataset for the target model to label: We've established that models will normally provide outputs even if the inputs aren't relevant. But when it comes to extracting a model, it's important to understand that you aren't extracting the exact model, you are simply modeling a model. Suppose you had access to a random 50% of the training data - how accurate would your copy-cat model be relative to the target model (trained on 100%)? Probably not as good without some careful modeling, and how would you know anyway? Or what do you think a copy-cat model would learn about other classes if you only had training data for a 7? Find a dataset that will give you accurate insights for the task you want to perform.

3. A model architecture that is useful and relevant to creating the outputs we want: You probably don't need the latest and greatest architecture, and you definitely don't need ChatGPT (though it can be used to collect a dataset - Alpaca style). Understand that algorithms are tools, each with its own purpose and reason for existing. Mathematics even has a theorem for this concept - there is "no free lunch". It's the idea that there is no one-size-fits-all algorithm or optimization strategy that works best for every problem. The "no free lunch" theorem states that any two optimization algorithms are equivalent when their performance is averaged across all possible problems. This means that there is no one algorithm that can outperform all others on every problem. Instead, the most effective algorithm for a particular problem depends on the specific characteristics of that problem. So like, experiment. A lot.

To do this we will send samples to the target model and collect outputs, then store them as a new dataset. We'll randomly sample images from the representative dataset, and we will cap target model queries with a query_budget constraint. This dataset will be what we use to train the copy-cat. We will create a custom torch Dataset, which is a fairly straight forward process. A torch Dataset is simply a Python class that inherits from torch.utils.data import Dataset, and implements __init__, __len__, and __getitem__.

# Model Extraction

- Do you want to know what the model thinks about cats? Send it pictures of all kinds of different cats.
- Do you want to know what the model thinks about malware? Send it all different kinds of malware.
- Do you want to know what the model thinks about X? Send it ~X.
There are some higher level strategies about what from your dataset you send first. For example, you could send a starting input, then measure the distance between that sample and everything else and choose, let's say, the next sample that is furthest away. This would ensure you're not sending similar samples and only learning about those from the model. This is anecdotal, but it seems that a more diverse dataset generally leads a more diverse set of observations to learn from. But when running an extraction attack you should consider the types of outputs you can receive (or the heuristic measures you could put in place),

- The worst case scenario is that the target returns a binary value like 0 or 1 (which could mean execution or not).
- The best case is scenario is the target product returns a continuous value, a value between 0 and 1 (or probabilities)

You will start to get a feel for the process and you analyze more and more datasets, but a lot of EDA (and model building) is cleaning data, and figuring out what's worth your time to analyze.

Once you start transforming data, try to write idempotent code. That's a fancy word that means "no matter how many times you execute this, you'll always get the same result." For example, we could have written the drop with something like df.drop(columns=["phishscore", "imposterscore"]), but after you've executed that code once any subsequent execution would result in an error (because those columns wouldn't exist anymore). Instead, by writing functional and idempotent code, we can execute the above cell any number of times while maintaining correct output. Idempotent data transformations will save you a lot of time and headaches.

In the next cell, we define a term-frequency * inverse document frequency (tfidf) vectorizer. This looks at how frequently terms occur and punishes things that occur too often (because they don't carry information that help with our machine learning task). This scikit-learn function also has some handy arguments to reduce our feature space like removing stop words and words that don't occur more than a certain frequency. Note that we didn't tokenize first, so "learn" and "learning" will be vectorized separately.

# Assessments
New concerns
- Ethics and safety - Trustworthy ML/Responsible AI
- Not the same as other technologies - Responsible use of Win32
- At a technical level, it's kind of all the same. What outcome do you want to optimize for?
    - A security issue?
    - A safety issue?
    - A trust issue?
- Where is your organization today?
    - If they have no idea, maybe you need consultations.
    - These risk may be new to them
    - What tools and skills do you already have?
- There are likely lots of issues, help develop priorities
- Security controls impact velocity + Many AI/ML research ideas never make it to production

ML stated at "red teaming", but still haven't done some of the most basic stuff:
- Pickles, eval/exec, no auth mechanisms
- Data collected straight from the internet and through straight into a model, with zero validation or controls
- ZERO pre-deployment testing (even ChatGPT)
- Red Teaming LLMs is popular, but to what end?
- Logging and monitoring on models? Nope

Tools!
- Adversarial Robustness Toolbox
- TextAttack
- Alibi

Wrap the predict function!
```
def predict(x):
    resp = "https://"

    resp.json()["labels"]
    return output
```

As a reminder (from the extraction lab), inputs is a dictionary of mutiple arrays that make up a single sample. This is a key difference between image classifiers amd text models or generative models.

The keys represents:

input_ids: These are the tokenized representations of the input text. Each integer represents a specific token in the model's vocabulary. For instance, in your example, the input_ids tensor includes the tokens for a specific sentence. The numbers 101 and 102 are special tokens, representing the start ([CLS]) and end ([SEP]) of a sequence, respectively.
token_type_ids: These are used in models that need to understand two separate sentences and how they interact (for example, in a question-answering model or a next-sentence prediction model). They differentiate between the two sentences. In this case, all the token_type_ids are 0, meaning that all the tokens belong to the same sentence. If there were two sentences, you'd see a sequence of 0s (for the first sentence) followed by a sequence of 1s (for the second sentence).
attention_mask: This is used to specify which tokens should be attended to by the model, and which should be ignored. A value of 1 means the token should be attended to, and a value of 0 means the token should be ignored. This is usually used when we have padded sequences (to make all inputs the same length) and we want the model to ignore the padding tokens. In your example, all values in the attention_mask tensor are 1, which indicates that all tokens in input_ids should be considered by the model.
These "extra" inputs are effectively just a way for us to provide a model contextual information. But they also represent an attack surface!

### Text Attack
Like ART, TextAttack requires input data to be structured a particular way, a list of tuples where each tuple consists of (input, label) and stored in a Dataset class, which is mostly just a container.

Instead of the predict function that we used in ART, TextAttack uses the __call__ method to query a model. Otherwise, the mechanics are the same.

Here, we can see the functions the attack is comprised of, this will also print when we run the attack. Faithful to their documentation, we can review the four primary components of an attack module (these should all be familiar sounding)

- Goal function: The goal function determines if the attack is successful or not. One common goal function is untargeted classification, where the attack tries to perturb an input to change its classification.
- Search method: The search method explores the space of potential transformations and tries to locate a successful perturbation. Greedy search, beam search, and brute-force search are all examples of search methods.
- Transformation: A transformation takes a text input and transforms it, for example replacing words or phrases with similar ones, while trying not to change the meaning. Paraphrase and synonym substitution are two broad classes of transformations.
- Constraints: Finally, constraints determine whether or not a given transformation is valid. Transformations don’t perfectly preserve syntax or semantics, so additional constraints can increase the probability that these qualities are preserved. This might include constraints that - overlap constraints that measure edit distance, syntactical constraints check part-of-speech and grammar errors, and semantic constraints. Or even execution contraints that we explored in the malware evasion lab.

### Alibi
Sample code:
```
img = np.moveaxis(unnormalize(img_tensor[0]).numpy(), 0, 2)

explainer = AnchorImage(
    predict, 
    image_shape=img.shape,
    segmentation_fn='slic',
    segmentation_kwargs={'n_segments': 30, 'compactness': 50, 'sigma': .1, 'start_label': 0,
                        'channel_axis':-1},
    images_background=None
)
```
The goal is to understand why a machine learning model makes a particular prediction on an image. To do this, AnchorImage breaks the image up into pieces called superpixels - think of these as little puzzle pieces of the image. It then starts removing or hiding different superpixel pieces of the image, and sees how much that changes the model's prediction.

The superpixel pieces that change the prediction a lot when removed are considered highly "influential" - meaning they contain visual patterns that the model relies on for the prediction. AnchorImage identifies the top most influential superpixel pieces. These influential pieces are called the "anchors" - they anchor the prediction locally to the image. By highlighting these anchor superpixels, we can visualize and understand the patterns in the image that drive the model's decision.

The segmentation_kwargs:

- The image is segmented into 30 superpixels, controlled by n_segments.
- compactness and sigma affect how cleanly the superpixels are divided.
- start_label=0 just means the first label for a superpixel is 0.
- No background images are provided, so influence is calculated against a blank background.

- Optuna (as an attack)


# Module 5 - Inversion
Inversion is the process of inverting a model in order to get a representation of the training data. Sometimes this is a 1:1 match for what the training data is, however, that is rare. It's best to imagine a model as a lossy compression algorithm. It contains a portion of the training data that it was trained on, however the trick is figuring out how to extract it and in practice, this is actually quite a difficult attack to execute. Not only does it take a ton of queries, but you never actually get training data - only a representation of it... at least for images. 
These attacks don't always work all the time.
- Although they're cooler attacks, they're not always effective, and sometimes hard.
Inversion is searching for a representation of the training data.

### Manual Inversion
1. Start with anything.
    - Solid color image
    - Random noise
    - Etc
2. Gradient descent to optimize the image 

#### MIFace
The MIFace attack was originally developed for facial recognition systems, where an image of a person that the classifier was trained on could be recovered. In practice, this works best for models (like facial recognition) where the examples for each class look very similar to each other. In ML terms, model inversion works best when the model is 'overfit' to a particular class, meaning it has effectively memorized it.

MIFace is an attack which allows us to invert a model that takes in an image and outputs a label. The attack itself is very similar to what we did above: we specify a model and an output and train the input to match that output. But just like training a model, the attack calculates gradients in order to determine how best to modify the input in order to provide the best output. As you may be noticing this also means that the attack requires access to the model's weights and won't work on a black box.

MIFace Template script:
```
classifier = PyTorchClassifier(model=model, 
                                # YOUR CODE HERE!
                              )

y = torch.tensor([0,1,2,3,4,5,6,7,8,9])

attack = MIFace(classifier, 
                # YOUR CODE HERE!
               )

x_train_infer = # YOUR CODE HERE! Make sure your final shape is (10, 1, 28, 28)

x_train_infer = attack.infer(x=x_train_infer, y=y)


# plotting boilerplate
fig = plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train_infer[i,0], cmap='gray')
```

MIFace Example Script:
```
classifier = PyTorchClassifier(model=model, 
                                clip_values=[0,1],
                               loss=F.cross_entropy,
                               input_shape=(1,1,28,28),
                               nb_classes=10
                              )

y = torch.tensor([0,1,2,3,4,5,6,7,8,9])

attack = MIFace(classifier, 
                max_iter=100,
                learning_rate = 0.1
               )

x_train_infer = np.zeros((10,1,28,28))

x_train_infer = attack.infer(x=x_train_infer, y=y)


# plotting boilerplate
fig = plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train_infer[i,0], cmap='gray')
```


The general method of attack for all these issues is the same, and should be a bit reminiscent of generating an adversarial example. If we think that the model is overfit on a specific image, then it should respond with a very strong classification for that input. If we hunt through the space of images to find one that elicits a strong repsonse from the model, then there's a good chance that maps to a training image.

Membership inference

The pytorch `state_dict` produces a dictionary where keys are parameter names, and values are the values of those parameters. These can be inspected for a model via model.state_dict() and loaded into a model (with checks to make sure that the keys match between the model and the dictionary being loaded) via model.load_state_dict().
```
# provided code

x = nn.Parameter(torch.ones(1,1,28,28)*.5)
x_optimizer = torch.optim.Adam([x]) # change this (some code in the torch lab might be helpful here

target = torch.tensor([0])
```

# Module 6 - Poisoning
From a Rules of Engagement stance, poisioning data can be dangerous, and not to be tested lightly (think FSD).
Takes less than 1% to poison data.

More than some other attacks, poisoning takes scale
- More data
- More epochs
- More time
- Potentially more access
Depending on your visibility into the training process, your data will probably be modified before used in training

The canonical training time attack is data poisoning. Poisoning is an attack that requires access to training data. This could be through network access or some other vector. The end goal is basically to influence the training of the model such that you gain a desired outcome some high percentage of the time.

Below is a way that someone may load the MNIST data. In PyTorch, a DataLoader contains the same functions you might see in a production Extract, Transform, Load (ETL) pipeline. The data is extracted from a web host or local directory, it's transformed using any functions you provide (in this case, converting to tensors), and loaded into batches and an iterator. In a less trivial example, the data may be extracted from a data warehouse, transformed by enriching it with other data sources, and loaded into a pub/sub queue.

#### Witch's Brew
Witches Brew is a poisoning attack that came out in 2021. It targets the reclassification of an unmodified test image. From the paper: "the central mechanism of the new attack is matching the gradient direction of the malicious sample." Basically, we're building an attack around the fact that gradient descent is a crucial part of training. We structure the training problem so that reducing training loss also reduces adversarial loss (a loss function mapping the success of our adversarial attack) -- aligning their gradients.

Witches Brew is more efficient and robust than preexisting attacks under similar constraints. "The attack successfully compromises a ResNet-34 by manipulating 0.1% of the data points with perturbations less than 8 pixel values in linf-norm... Strong differential privacy is the only existing defense that can partially mitigate the effects of the attack."

Tradecraft
Witches Brew has some nice tradecraft properties:

Clean Label: This is a class of "clean-label" attacks, meaning that while we can modify data bits, we don't modify the labels. This is an important element of tradecraft; Witches Brew is a "clean label" attack. The easiest way to solve the MNIST exercise above is by modifying image labels, but that is easily detected with human supervision and inspection. By contrast, a clean label attack doesn't change the image labels and therefore can bypass manual human inspection.
Survives Data Augmentation: In many ML contexts, each training datapoint is augmented to improve the robustness of the final model. Images might be cropped, rotated, flipped, and skewed, for example. Witches Brew is relatively robust to these augmentations because it performs data augmentation for each poisoned image. This also improves the transferability of the attack between models.
Does not require perfect knowledge of training procedures: The attacker does not need to know about the victim's model initialization, batches, or data augmentation steps.
Does not require access to all training data: Usually requires less than 1% of the training data and no access to validation data.
No apparent impact to training/validation accuracy: Reduces the chances of detection during training or potential accuracy degredations launching promotions to production.
Successful when model trained from scratch or finetuned: Previous methods (like Poison Frogs) only work in finetuning settings.
Gradient matching is efficient: While it does require a set of trained parameters, generating poisoned images doesn't require training a classifier. This reduces attack cost significantly. "Poisoning ImageNet for a ResNet-18 with MetaPoison would take 500 GPU days vs 4 GPU hours for Witches Brew."

## Persistance
Poor security controls around model persistance is a frequent opportunity
Charcuterie

Pickles are still very persistant and dangerous
1. Deserialzing untrusted data is extremely dangerous
2. This is a well know issue across languages
3. Most "solutions" are ultimately inaffective
    - Blocklisting
    - Allowlisting
Rearchitecting should be the only goal (safetensors, ONNX)
HuggingFace has a Pickle scanner
https://github.com/protectai/modelscan

### ONNX - https://github.com/onnx/onnx
ONNX
ONNX was developed as an attempt to standardize model files into a format that could be serialized and deserialized by a variety of libraries. You'll find ONNX methods exposed by many common frameworks. There's also a fairly comprehensive model zoo that might come in handy.

Exploiting Onnx is not nearly as simple as pickled formats, however, depending on your goals sometimes you can find interesting operations exposed to ONNX by a library which can be abused or exploited. And when custom opperations are added there may be exploits relating to DLL side loading or by exploiting the logic of the operation itself.

### TensorFlow
https://www.tensorflow.org/tutorials/keras/save_and_load

By default, Keras uses HDF5. In contexts where the model may be deployed beyond Python (to TFLite or TensorFlow.js, for example), you're likely to find SavedModel objects. The underlying serialization format is protobuf. It's unlikely that this will lead to a compromise baring some other issue, however this also means the models control logic is not part of the format. This may mean it's shared between applications in one of three ways.

Infered from the object
Shared as code
It's static
The third option is likely not going to be a point of failure. However if there is an issue, the seccond would be a fairly standard build pipeline attack most likely, and the first would be an application logic attack.

Tensorflow also suports something called lambda layers which are saved as python byte code. This should not be a reliable way to load the model on a different system. However if one is saved somewhere an attacker can control, and loaded by a system they don't have access to, it could lead to a compromise.


# Module 7: LLMs

## Finetuning
Models receive all words at once today.
Having the entire text available for analsis helps the model decode ambiguous words and resolve references.
Models have multiple attention heads (read heads)

Sampling
- temp - rescale the logits to balance distribution
- sample - multinomial sample on the distribution
- top_k - clip distribution to only those top tokens (e.g. top_k=2)
- beam - step beyond 1 toek to find high prob seq
- top_p - clip distribution based on cumulative probs
https://huggingface.co/blog/how-to-generate

Should you even train? Maybe not.
- Foundation models
- Instruction-tuned models

## Prompting
## Poisoning
## Data Extraction


# Relevant Links:
- http://websocketstest.courses.nvidia.com - Check on the browser and plugins for compatibility
- learn.nvidia.com - main site
- http://matrixmultiplication.xyz - for reviewing matrix multiplication
- https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
- https://arxiv.org/abs/1905.07121
- https://crfm.stanford.edu/2023/03/13/alpaca.html
- https://developer.nvidia.com/blog/nvidia-ai-red-team-an-introduction
- https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/estimators/classification.html#pytorch-classifier
- https://arxiv.org/abs/2302.10149
- https://github.com/moohax/Charcuterie
- https://github.com/protectai/modelscan
- https://github.com/onnx/onnx
- https://www.tensorflow.org/tutorials/keras/save_and_load
- https://proceedings.neurips.cc/paper/2020/file/8ce6fc704072e351679ac97d4a985574-Paper.pdf
- https://huggingface.co/blog/how-to-generate


## Lab: Basic Modeling
1. Explore data structures and some reshaping
2. Introduce linear models
3. Introduce linear models + evaluations

## (Pytorch) How to install

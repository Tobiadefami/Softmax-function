# A Subtle Introduction to the Softmax Function

The softmax function is a generalization of the logistic function to multiple dimensions. It is often used as the last activation function of a neural network to normalize the output of a network and predict the probabilities associated with a multinoulli distribution.

## The Softmax Function

The softmax function takes an input vector **a** of $X$ real numbers, and normalizes it into a probability distribution consisting of $X$ probabilities proportional to the exponentials of the input numbers, this exponentiation forces the output to be positive numbers. 

Essentially this means that before applying the softmax function, some vector components could possess negative properties, be greater than one; and might not sum to 1; but on the application of softmax, each component will be in the interval $[0, 1]$ and the components will add up to $1$, so that they can be interpreted as probabilities. Furthermore, the larger input components will correspond to larger probabilities. 

#### Let's introduce some interesting mathematics: 

The standard softmax function $\sigma : \mathbb R^X \to [0,1]^X$ is defined by the formula:

$$\sigma(\bold{a})_i = \frac{e^{a_i}}{\sum_{j=1}^Xe_j^a}\;\;\; \forall \;i\in\{1,X\} \;\text{and} \; \bold{a} = (a_i, ..., a_X) \in \mathbb R^X \tag{1}$$  

In plain English, it applies the standard exponential function to each element $a_i$ of the input vector **a** and normalizes these values by dividing by the sum of all these exponentials; this normalization ensures that the sum of the components of the output vector $\sigma(\bold{a})$ is 1. 


Instead of assigning the base to Euler's number, a different base could be used, provided the base is greater than zero. Say we assign a base $b>0$:

1. For $0<b<1$, the softmax function returns larger output probabilities for smaller input components, and decreasing the value of $b$ creates probability distributions that are centralized around the positions of the smallest input values.
    
2. If $b>1$, larger input components will result in larger output probabilities, and increasing the value of $b$ creates probability distributions that are centralized around the positions of the largest input values. 

&nbsp;

Okay, enough with the boring stuff, let's jump into the cool stuff 😎 and show how we can implement this activation function using our friendly neighbourhood programming language, python 😂

We can write equation (1) as a python function illustrated in the code block given below.

```python
import numpy as np

def smax(x):
  for i in range(1, len(x)):
    for j in range(1, len(x)):
      den = np.sum(np.exp(x))
      num = np.exp(x)
  return num/den

#test the function
arr = np.array([4,2,3])
smax(arr) 
# Returns
>>> array([0.66524096, 0.09003057, 0.24472847])

np.sum(smax(arr))
>>> 1.0 # softmax condition satisfied
```


Luckily, python has amazing libraries that help with the computation of this function, eliminating the need to write bespoke functions every time we need to make this computation as we did above. Using libraries like [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html?highlight=softmax#torch.nn.Softmax) and [Tensorflow/Keras](https://keras.io/api/layers/activation_layers/softmax/), we can compute the softmax function in just a few lines of code.


**With Pytorch**
```python
import torch
import numpy as np

arr = np.array([4.,2.,3.])

m = torch.nn.Softmax()
input = torch.from_numpy(arr)
output = m(input)

print(output)

# Returns
>>> tensor([0.6652, 0.0900, 0.2447], dtype=torch.float64)
```
**With Tensorflow/Keras**

```python
import tensorflow as tf
import numpy as np

arr = np.array([4.,2.,3.])
layer = tf.keras.layers.Softmax()
layer(arr).numpy()

# Returns
>>> array([0.6652409 , 0.09003057, 0.24472845], dtype=float32)
```
&nbsp;

**Conclusion**
In this article, we introduced the softmax function, defined a mathematical representation for it, and showed how to implement it using python. Hope you enjoyed it, see you next time!


&nbsp;

**References**
1. Softmax function https://en.wikipedia.org/wiki/Softmax_function

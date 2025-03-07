# neural-network-from-scratch
A single layer neural network from scratch for a binary classification problem.

For the single layer MLP, the input undergoes the following transformations. Firstly, the column vector $x$ goes through a linear transformation:

$$\
q = W^1x + b^1
\$$

Then non-linearity is added using an element-wise ReLU activation function:

\$$
h = \text{ReLU}(q) = \max(0, q)
\$$

This output is then fed to the next linear transformation to get the logits \( o \):

\$$
o = W^2h + b^2
\$$

Lastly, applying softmax on the logits yields the probability distribution over the classes \( p \):

\$$
p = \text{softmax}(o)
\$$

The cross entropy loss function is defined as:

\$$
L(p, y) = -\sum_{i=1}^{M} y_i \log(p_i)
\$$

where $M$ is the total number of classes.

# Introduction

THIS IS VERY MUCH A WORK IN PROGRESS, BUT HOPEFULLY IS INTERESTING NONETHELESS!!

This is a repository for alternative learning rules to backpropagation. Most of the implemented methods use no backpropagation, however some do use it, but are not completely reliant on it (fast weight programmers and reservoir computers).

The following methods are implemented:

* Hebbian Learning
* Predictive coding
* [Fast Weight Programmers](https://arxiv.org/abs/1610.06258)
* [Generalized principal subspace projection](https://arxiv.org/abs/2302.10051)
* Reservoir Computing
* Genetic algorithms
* Forward only learning ([Forward-forward Learning](https://arxiv.org/abs/2212.13345) and [PEPITA](https://arxiv.org/pdf/2201.11665))

In general, the rules are implemented and tested on MNIST.

# Results

All of the fully implemented algorithms are compared by their sample efficiency (how accurate are they with N samples?) and clock time efficiency (how accurate are they with X seconds of train time?). Architectures were identical as possible, however there are some caveats to that:

* Hebbian learning requires odd activation functions so Tanh was used, while backprop-based training worked very poorly with Tanh, so in fairness it was replace with ReLU
* The bidirectional version of predictive coding has forward and backward layers, so it technically has double the parameters in training, however a inference forward pass still uses the same number of parameters
* Reservoir computing involves a blob of neurons, I did some arithmetic so the blobs have equal synapse/parameter count as the layered networks

The following plot shows sample efficiency:

![Sample Efficiency](plots/Sample%20Efficiency.png)

Backprop and the reservoir computer required optimizers, SGD was used with learning rate 0.01. Note that forward-forward does not function well with continuous (i.e. in [0, 1] values). The paper uses binary pixel values, which I did not do for comparison with the other backpropagation alternatives.

Sample efficiency is not the whole story: predictive coding relies on a slow equilbration phase involving hundreds of forward passes per sample. For this reason, time taken is also important:

![Clock Time](plots/Clock%20Time.png)


## TODOs

Checked means started, not done (it wouldn't be on here if it was done!). Unchecked means not started.

- [x] Finish RL with dynamical systems
- [x] Finish genetic algorithms
- [x] Finish empowerment [implementation](https://arxiv.org/abs/1710.05101)
- [ ] Add generec and pineida-almeida
- [ ] more plots, including depth (also store everything in pandas)

# Implementation Overview

## Hebbian Learning

Hebbian learning is a classic theory of neural network learning, inspired by neuroscience. In the most simple case, neural networks update with the rule "neurons that fire together, wire together" i.e. neurons update their weights based on the correlation between input and output. This turns out to essentially do a computation of low-rank version of the principal component matrix. There are many more variants of Hebbian learning, which capture things like nonlinearities, regularization, and clustering.

Here, the variant of Hebbian learning that is like "independent component analysis" is implemented as well as the "soft winner-take-all" method (which has also gone under the name "competitivie learning"). Hebbian learning was implemented using techniques from several papers:

* [FastHebb](https://arxiv.org/abs/2207.03172) is implemented, this is a refactoring of Hebbian learning for speed in GPU implementations
* Several methods for "nonlinear PCA" are implemented [1](https://ieeexplore.ieee.org/document/374363) [2](https://citeseerx.ist.psu.edu/viewdoc/download?repid=rep1&type=pdf&doi=10.1.1.38.8171) [3](https://is.mpg.de/fileadmin/user_upload/files/publications/pdf2302.pdf) [4](http://www.scholarpedia.org/article/BCM_theory)

Implementing [convolutional Hebbian learning](https://openportal.isti.cnr.it/doc?id=people______::c8f9c1662c164f852a87b32d6d6bb3e1) is a to-do.

## Predictive Coding

Predictive coding is a neuroscience-inspired learning method that avoids some of the implausibility of backpropagation without sacrificing global error signals. The essential step is that only errors from predictions are passed "upwards" in the neural network. The error is calulated either via comparison with a backward prediction (in the bidrectional case) or via a energy-based relaxation phase that follows an energy gradient that decreases the errors then updates the weights to make that state more likely in the future (in feedforward case, see the linked paper for details). Note that predictive coding often has many Hebbian components, but it's listed seperately because there are major differences and non-Hebbian aspects to the learning.

Three variants are implemented here (with some subvariants):

* Bidirectional predictive coding (largely based on [this book](https://mitpress.mit.edu/9780262545617/gradient-expectations/))
* [Feedforward predictive coding](https://pubmed.ncbi.nlm.nih.gov/28333583/)
* [Incremental PC (iPC)](https://openreview.net/forum?id=RyUvzda8GH)

## Generalized Principal Subspace Projection (GSSP)
This is based on a [recent paper](https://arxiv.org/abs/2302.10051) from Flatiron Institute. They derive online, non-Hebbian learning rules based on the idea of pyramidal neurons in the cerebral cortex being multicompartmental: they have distinct (apical, proximal) dendrites that integrate information from discrete neural populations. Therefore, this rule is based on learning involving integrating two seperate multidimensional signals.

## Fast Weight Programmers

TODO!

## Reservoir Computing

I was inspired by [this paper](https://arxiv.org/abs/2210.10211) to implement reservoir computing as well. In reservoir computing, a large randomly connected (not generally feedforward!) neural network (the reservoir) is initialized. In general, these neural networks are highly recurrent. They have read-in weights, which are channel by which data can causally influence the network, and state update weights which describe how the neural dynamical system evolves with time. These weights are not updated with any learning rules.

The reservoir is essentially doing a nonlinear signal analysis of the data. In order to parse this, a linear readout network is trained on the reservoir state (so that in e.g. MNIST it is predicting class from the reservoir state). In the ideal case, the reservoir seperates signals in the data so that the simple linear readout can make strong predictions. The linear readout network here can either be trained with backprop (default) or Hebbian WTA learning.

## Genetic algorithms

TODO! There are some preliminary implementations of CMA-ES and a custom GA, but they need work.

## Forward-only Learning

This implements the forward-forward algorithm by Hinton, based on his [Matlab source](https://www.cs.toronto.edu/~hinton/ffcode.zip) (note, this link is a zip file that will download). Additionally, I implemented [layer collaboration based on this paper](https://arxiv.org/abs/2305.12393), which partially ameliorates the greedy layerwise aspect of the algorithm.

The forward-forward algorithm is essentially a contrastive learning framework. In the image + label case, images are concatenated to their labels (since the learning method is not supervised, so concatenation learns the joint distribution in an unsupervised way). Next, a negative sample is generated by utilizing the same image, but concatenating a random label. The network is then trained to modify its weights by minimizing (maximizing) an energy function for the negative (positive) samples using a layer-local gradient update.

Unlike the original/official implementation, I do not force images to be binary, pixel values are treated as continuous numbers.

The [PEPITA method](https://arxiv.org/pdf/2201.11665) of forward only learning is also implemented.

# Other Notes

I've also implemented some interesting papers with graph theoretic tools that I thought might help analyze the backprop alternatives. A potentially incomplete list:

* Various graph/network hierarchy measures which give indicators of the hierarchy in a graph based on directed information flow, from [this paper](https://www.nature.com/articles/s41598-021-93161-4).
* [Laplacian renormalization](https://www.nature.com/articles/s41567-022-01866-8), a recent graph theoretic analogue to renormalization methods in statistical mechanics. In particular, this enables scaling networks while preserving certain key properties.
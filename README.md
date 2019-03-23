# Learn sparsity pattern
Magnitude-based pruning is widely used in network compression, which
simply prunes away weights below certain manually set magnitude threshold.
But the magnitude of weights doesn't mean importance necessarily, because
there are some correlations between weights like sharpening filter. This
project aims to study the correlation between weights using a hypernetwork
and predict the importance of weights and filters.



## Methods
* l1-regularization(baseline): add l1-regularization on weights
* dns(baseline): prune weights under certain threshold and splice dynamically
* mask network: learn a hypernetwork to predict a binary mask

The input of mask network is conv2 weights of each layer, which is fed
through a convolutional neural network followed by binarization function
to produce a binary mask. By using some approximation techniques, this
framework is differentiable and end-to-end trainable.

## Parameter setting
Good paractice of parameter setting to preserve accuracy while
maximizing sparsity

parameters     | l1-reg | dns  | mask network
-------------- | ------ | ---- | ----------- |
reg_lambda     | 1e-2   |      | 1e-5
dns_threshold  |        | 1e-1 |
l1_threshold   | 1e-4   |      |

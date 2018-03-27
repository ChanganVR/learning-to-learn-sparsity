# Learn sparsity pattern
Learn to sparsity a deep neural network via learning a meta/hyper-netowrk
to predict the importance of weights


## Methods
* l1-regularization(baseline): add l1-regularization on weights
* dns(baseline): prune weights under certain threshold and splice dynamically
* mask network: learn a hypernetwork to predict a binary mask

## Parameter setting
Good paractice of parameter setting to preserve accuracy while
maximizing sparsity

parameters     | l1-reg | dns  | mask network
-------------- | ------ | ---- | ----------- |
reg_lambda     | 1e-2   |      | 1e-5
dns_threshold  |        | 1e-1 |
l1_threshold   | 1e-4   |      |
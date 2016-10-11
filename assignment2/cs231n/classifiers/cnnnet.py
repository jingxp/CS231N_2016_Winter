import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A n-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32,48,64], filter_size=[5,3,3],
               hidden_dim=[500, 100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters[0], C, filter_size[0], filter_size[0])
    self.params['b1'] = np.zeros((1, num_filters[0]))
    self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1])
    self.params['b2'] = np.zeros((1, num_filters[1]))
    self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2])
    self.params['b3'] = np.zeros((1, num_filters[2]))
 
    H3=num_filters[2]*H*W/64
    self.params['W4'] = weight_scale * np.random.randn(H3, hidden_dim[0])
    self.params['W5'] = weight_scale * np.random.randn(hidden_dim[0],hidden_dim[1])
    self.params['W6'] = weight_scale * np.random.randn(hidden_dim[1], num_classes)
    self.params['b4'] = np.zeros((hidden_dim[0]))
    self.params['b5'] = np.zeros((hidden_dim[1]))
    self.params['b6'] = np.zeros((num_classes))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size1 = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}

    filter_size2 = W2.shape[2]
    conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}

    filter_size3 = W3.shape[2]
    conv_param3 = {'stride': 1, 'pad': (filter_size3 - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_param2 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    pool_param3 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_forward_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param1, pool_param1)
    out_forward_2, cache_forward_2 = conv_relu_pool_forward(out_forward_1, self.params['W2'], self.params['b2'], conv_param2, pool_param2)
    out_forward_3, cache_forward_3 = conv_relu_pool_forward(out_forward_2, self.params['W3'], self.params['b3'], conv_param3, pool_param3)
    out_forward_4, cache_forward_4 = affine_relu_forward(out_forward_3, self.params['W4'], self.params['b4'])
    out_forward_5, cache_forward_5 = affine_relu_forward(out_forward_4, self.params['W5'], self.params['b5'])
    scores, cache_forward_6 = affine_forward(out_forward_5, self.params['W6'], self.params['b6'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y) 
    da5, dW6, db6 = affine_backward(dscores, cache_forward_6)
    da4, dW5, db5 = affine_relu_backward(da5, cache_forward_5)
    da3, dW4, db4 = affine_relu_backward(da4, cache_forward_4)
    da2, dW3, db3 = conv_relu_pool_backward(da3, cache_forward_3)
    da1, dW2, db2 = conv_relu_pool_backward(da2, cache_forward_2)
    dX, dW1, db1 = conv_relu_pool_backward(da1, cache_forward_1)

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5
    dW6 += self.reg * W6

    reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6])

    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5,  'W6': dW6, 'b6': db6}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

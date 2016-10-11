import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  buf_e = np.zeros(num_classes)
  for i in xrange(num_train):
    for j in xrange(num_classes):
        buf_e[j]=X[i,:].dot(W[:,j])
    buf_e-=np.max(buf_e)
    buf_e=np.exp(buf_e)
    buf=buf_e/np.sum(buf_e)
        
    loss -= np.log(buf[y[i]])
    for j in xrange(num_classes):
        dW[:,j] +=( buf[j] - (j ==y[i]) )*X[i,:].T 
     
  loss /= num_train
  dW /= num_train
  
  loss += 0.5 * reg * np.sum(W * W)
  dW +=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  scores=np.subtract(scores.T,np.max(scores,axis=1)).T
  scores=np.exp(scores)
  scores=np.divide(scores.T,np.sum(scores,axis=1).T).T

  loss = - np.sum(np.log ( scores[np.arange(num_train),y] ) )
  loss /=num_train  + 0.5 * reg * np.sum(W * W)
  loss += 0.5 * reg * np.sum(W * W)
  scores[np.arange(num_train),y]  -= 1
  dW = np.dot(X.T,scores)/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


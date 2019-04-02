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
  pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  #To avoid numerical instability
  score_max = np.max(scores, axis = 1, keepdims = True)
  scores -= score_max
  scores = np.exp(scores)
  #calculating log probabilitys
  for i in range(num_train):
    sum_per_example = np.sum(scores[i,:])
    p = lambda value : value/sum_per_example
    loss -= np.log(p(scores[i,y[i]]))
    
    for j in range(num_classes):
        p_j = p(scores[i,j])
        dW[:,j] += (p_j  - (j == y[i]))*X[i]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)
  dW /= num_train
  dW += reg * W  
    
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
  pass
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  #To avoid numerical instability
  score_max = np.max(scores, axis = 1, keepdims = True)
  scores -= score_max
  scores = np.exp(scores)
  sum_s = np.sum(scores, axis = 1, keepdims = True)
  probability = scores/sum_s
  loss = -np.sum(np.log(probability[np.arange(num_train), y]))
  temp = np.zeros_like(probability)
  temp[np.arange(num_train), y] = 1
  dW = X.T.dot(probability-temp)
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  max_margin = 0.0

  for i in xrange(num_train):
      scores = W.dot(X[:, i])
      scores -= np.max(scores)
      normalized = np.exp(scores) / np.sum(np.exp(scores))
      loss -= np.log(normalized[y[i]])


      # for j in xrange(num_classes):
      #     if j == y[i]:
      #         continue
      #     cross_entropy_loss =
      #     if margin > 0:
      #         max_margin -= np.max(margin)
      #         loss = np.exp(max_margin)/np.sum(np.exp(max_margin))
              # Compute gradients (one inner and one outer sum)
              # Wonderfully compact and hard to read
              # dW[y[i], :] -= X[:, i].T  # this is really a sum over j != y_i
              # dW[j, :] += X[:, i].T  # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Same with gradient
  # dW /= num_train

  # Add regularization
  # loss += 0.5 * reg * np.sum(W * W)

  # Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
  # dW += reg * W
  
  # return loss, dW
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################





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
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

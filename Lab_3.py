import numpy as np
np.random.seed(1)

def sigmoid(z):
  """
  sigmoid function that maps inputs into the interval [0,1]
  Your implementation must be able to handle the case when z is a vector (see unit test)
  Inputs:
  - z: a scalar (real number) or a vector
  Outputs:
  - trans_z: the same shape as z, with sigmoid applied to each element of z
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  trans_z = 1.0 / (1.0 + np.exp(-z))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return trans_z

def logistic_regression(X, w):
  """
  logistic regression model that outputs probabilities of positive examples
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  Outputs:
  - logits: a vector of shape (num_samples,)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  logits = sigmoid(np.matmul(X, w))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return logits


  def logistic_loss(X, w, y):
	  """
	  a function that compute the loss value for the given dataset (X, y) and parameter w;
	  It also returns the gradient of loss function w.r.t w
	  Here (X, y) can be a set of examples, not just one example.
	  Inputs:
	  - X: an array of shape (num_sample, num_features)
	  - w: an array of shape (num_features,)
	  - y: an array of shape (num_sample,), it is the ground truth label of data X
	  Output:
	  - loss: a scalar which is the value of loss function for the given data and parameters
	  - grad: an array of shape (num_featues,), the gradient of loss 
	  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  loss = 1/len(X) * (-1 * np.matmul(y, np.log(sigmoid(np.matmul(X, w)))) - np.matmul(np.ones(len(X)) - y, np.log(np.ones(len(X)) - sigmoid(np.matmul(X, w)))))
  grad = -1/len(X) * np.matmul(np.transpose(X), (y - sigmoid(np.matmul(X, w))))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad

  def softmax(x):
	  """
	  Convert logits for each possible outcomes to probability values.
	  In this function, we assume the input x is a 2D matrix of shape (num_sample, num_classes).
	  So we need to normalize each row by applying the softmax function.
	  Inputs:
	  - x: an array of shape (num_sample, num_classse) which contains the logits for each input
	  Outputs:
	  - probability: an array of shape (num_sample, num_classes) which contains the
	                 probability values of each class for each input
	  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  exp_matrix = np.exp(x)
  probability = np.transpose(np.transpose(exp_matrix) / np.matmul(exp_matrix, np.ones(len(x[0]))))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def MLR(X, W):
  """
  performs logistic regression on given inputs X
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  Outputs:
  - probability: an array of shape (num_sample, num_classes)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  probability = softmax(np.matmul(X, W))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

  def cross_entropy_loss(X, W, y):
	  """
	  Inputs:
	  - X: an array of shape (num_sample, num_feature)
	  - W: an array of shape (num_feature, num_class)
	  - y: an array of shape (num_sample,)
	  Ouputs:
	  - loss: a scalar which is the value of loss function for the given data and parameters
	  - grad: an array of shape (num_featues, num_class), the gradient of the loss function 
	  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  one_hot = np.zeros((y.size, len(W[0])))
  one_hot[np.arange(y.size), y] = 1

  loss = -1 * 1 / len(X) * np.sum(np.log(np.sum(MLR(X, W) * one_hot, axis=1)))
  grad = -1 / len(X) * np.matmul(np.transpose(X), one_hot - MLR(X, W))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad

  def gini_score(groups, classes):
	  '''
	  Inputs: 
	  groups: 2 lists of examples. Each example is a list, where the last element is the label.
	  classes: a list of different class labels (it's simply [0.0, 1.0] in this problem)
	  Outputs:
	  gini: gini score, a real number
	  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  # count all samples at split point
  from collections import defaultdict
  total_examples = 0
  gini_indexes = {}
  no_examples_per_group = {}
  for i, elem in enumerate(groups):
    total_examples += len(elem)
    no_examples_per_group[i] = len(elem)
    crt_mapper = defaultdict(int)
    for datapoint in elem:
      crt_mapper[datapoint[-1]] += 1
    s = 0

    for cls in crt_mapper:
      s += (crt_mapper[cls] / len(elem))**2

    gini_indexes[i] = 1 - s

  gini = 0
  for i in gini_indexes:
    gini += (no_examples_per_group[i] / total_examples) * gini_indexes[i]
  # sum weighted Gini index for each group

  # sum weighted Gini index for each group

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return gini

  def create_split(index, threshold, datalist):
	  '''
	  Inputs:
	  index: The index of the feature used to split data. It starts from 0.
	  threshold: The threshold for the given feature based on which to split the data.
	        If an example's feature value is < threshold, then it goes to the left group.
	        Otherwise (>= threshold), it goes to the right group.
	  datalist: A list of samples. 
	  Outputs:
	  left: List of samples
	  right: List of samples
	  '''
  
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  left = []
  right = []
  for elem in datalist:
    if elem[index] < threshold:
      left.append(elem)
    else:
      right.append(elem)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return left, right

  def get_best_split(datalist):
	  '''
	  Inputs:
	  datalist: A list of samples. Each sample is a list, the last element is the label.
	  Outputs:
	  node: A dictionary contains 3 key value pairs, such as: node = {'index': integer, 'value': float, 'groups': a tuple contains two lists of examples}
	  Pseudo-code:
	  for index in range(#feature): # index is the feature index
	    for example in datalist:
	      use create_split with (index, example[index]) to divide datalist into two groups
	      compute the Gini index for this division
	  construct a node with the (index, example[index], groups) that corresponds to the lowest Gini index
	  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  crt_classes = [0, 1]
  best_gini = 1

  for idx in range(len(datalist[0]) - 1):
    for example in datalist:
      left, right = create_split(idx, example[idx], datalist)
      gini = gini_score((left, right), crt_classes)
      if gini < best_gini:
        best_gini = gini
        best_idx = idx
        best_threshold = example[idx]
        best_left = left
        best_right = right

  node = {'index':best_idx, 'value':best_threshold, 'groups':(best_left, best_right)}

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return node

  def to_terminal(group):
	  '''
	  Input:
	    group: A list of examples. Each example is a list, whose last element is the label.
	  Output:
	    label: the label indicating the most common class value in the group
	  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  mapper = defaultdict(int)

  for elem in group:
    mapper[elem[-1]] += 1

  m = -1
  label = -1
  for idx in mapper:
    if mapper[idx] > m:
      m = mapper[idx]
      label = idx

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return label


  def recursive_split(node, max_depth, min_size, depth):
	  '''
	  Inputs:
	  node:  A dictionary contains 3 key value pairs, node = 
	         {'index': integer, 'value': float, 'groups': a tuple contains two lists fo samples}
	  max_depth: maximum depth of the tree, an integer
	  min_size: minimum size of a group, an integer
	  depth: tree depth for current node
	  Output:
	  no need to output anything, the input node should carry its own subtree once this function ternimate
	  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  if len(node['groups'][0]) == 0 or len(node['groups'][1]) == 0:
    if len(node['groups'][0]) == 0:
      label = to_terminal(node['groups'][1])
    else:
      label = to_terminal(node['groups'][0])
    
    node['left'] = label
    node['right'] = label
    node.pop('groups')
    return

  # check for max depth

  if depth >= max_depth - 1:
    node['left'] = to_terminal(node['groups'][0])
    node['right'] = to_terminal(node['groups'][1])
    node.pop('groups')
    return
  # process left child

  if len(node['groups'][0]) <= min_size:
    node['left'] = to_terminal(node['groups'][0])
  # process right child

  if len(node['groups'][1]) <= min_size:
    node['right'] = to_terminal(node['groups'][1])


  if 'left' not in node:
    new_node = get_best_split(node['groups'][0])
    node['left'] = new_node
    recursive_split(new_node, max_depth, min_size, depth + 1)

  if 'right' not in node:
    new_node = get_best_split(node['groups'][1])
    node['right'] = new_node
    recursive_split(new_node, max_depth, min_size, depth + 1)

  node.pop('groups')


  # check for a no split

  # check for max depth

  # process left child

  # process right child

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY 


  def build_tree(train, max_depth, min_size):
	  '''
	  Inputs:
	    - train: Training set, a list of examples. Each example is a list, whose last element is the label.
	    - max_depth: maximum depth of the tree, an integer (root has depth 1)
	    - min_size: minimum size of a group, an integer
	  Output:
	    - root: The root node, a recursive dictionary that should carry the whole tree
	  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  root = get_best_split(train)
  recursive_split(root, max_depth, min_size, 1)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return root

  def predict(root, sample):
	  '''
	  Inputs:
	  root: the root node of the tree. a recursive dictionary that carries the whole tree.
	  sample: a list
	  Outputs:
	  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  if isinstance(root, int):
    return root

  if sample[root['index']] < root['value']:
    return predict(root['left'], sample)
  else:
    return predict(root['right'], sample)

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

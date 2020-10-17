#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
from scipy.special import comb

#coding assignment
def cumulative_comb_with_repetition(n, k):
    """
    Compute the number of possible non-negative, integer solutions to
    x1 + x2 + ... + xk <= n.
    
    We will use this function to compute the dimension 
    of order k polynomial feature vector

    Args:
        n: integer, the number of "balls" or "stars"
        k: integer, the number of "urns" or "bars"
        
    Returns: the total number of combinations, integer.
    """
    # your code below
    combination = (comb((n+k),k, exact = False,repetition=False))
    # your code above
    return int(combination)
    #raise notImplementedError


# In[58]:


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code below
    f_x = np.dot(current_theta.transpose(), feature_vector)+current_theta_0
    if f_x <= 0:
        current_theta +=  label * feature_vector
        current_theta_0 +=  label
    return (current_theta, current_theta_0)
    
    # your code above
    #raise NotImplementedError


# In[59]:


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        np.random.seed(1)
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        return indices


# In[60]:



def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """


    current_theta = np.zeros((feature_matrix.shape[1],))
    current_theta_0 = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code below
             current_theta, current_theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],current_theta,current_theta_0)
            # Your code above
             #pass
    return current_theta, current_theta_0


# In[61]:


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code below
    z = np.dot(theta.transpose(), feature_vector)+theta_0
   
    if z >= 1:
        return 0
    else:
        return 1-z

    # your code above  
    #raise NotImplementedError


# In[62]:


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code below
    loss = 0
    for i in range(len(labels)):
        loss += hinge_loss_single(feature_matrix[i],labels,theta, theta_0)
        
    return loss/len(labels)
    # your code above 
    #raise NotImplementedError


# In[63]:


def gradient_descent(feature_matrix, label, learning_rate = 0.05, epoch = 1000):
    """
    Implement gradient descent algorithm for regression.
    
    Args:
        feature_matrix - A numpy matrix describing the given data, with ones added as the first column. Each row
        represents a single data point.
        
        label - The correct value of response variable, corresponding to feature_matrix.
        
        learning_rate - the learning rate with default value 0.5
        
        epoch - the number of iterations with default value 1000

    Returns: A numpy array for the final value of theta
    """
    n = len(label)
    theta = np.zeros(feature_matrix.shape[1])# initialize theta to be zero vector
    for i in range(epoch):
        # your code below
        y_cap = np.dot(feature_matrix,theta)
        error = (label-y_cap)
        # compute (average) gradient below
        average_gradient = (-1/n) * (error.T.dot(feature_matrix)) 
        # update theta below 
        theta = theta - learning_rate * average_gradient

        # compute the value of cost function
        # It is not necessary to comput cost here. But it is common to use cost 
        # in the termination condition of the loop
        cost = (error.T.dot(error))/(2*n)
        #stop iteration at convergence
        if cost <= 0.0001:
            break
        # your code above 
        # test
        # print(i, theta, cost)
    
  
        
    return cost,theta
    raise NotImplementedError


# In[64]:


def stochastic_gradient_descent(feature_matrix, label, learning_rate = 0.05, epoch = 1000):
    """
    Implement gradient descent algorithm for regression.
    
    Args:
        feature_matrix - A numpy matrix describing the given data, with ones added as the first column. Each row
        represents a single data point.
        
        label - The correct value of response variable, corresponding to feature_matrix.
        
        learning_rate - the learning rate with default value 0.5
        
        epoch - the number of iterations with default value 1000

    Returns: A numpy array for the final value of theta
    """
    n = len(label)
    theta = np.zeros(feature_matrix.shape[1])    # initialize theta to be zero vector

    for i in range(epoch):
        # your code below 
        # generate a random integer between 0 and n
        rand = np.random.randint(0,n)
        # compute gradient at this randomly selected feature vector below
        x = feature_matrix[rand, :]
        y = label[rand]
        y_cap = np.dot(x,theta)
        error = (label-y_cap)

        gradient =  (-1)*(error.T.dot(feature_matrix))
        
        # update theta below 
        theta = theta - learning_rate * gradient
        
        # compute average squared error or empirical risk or value of cost function
        # It is not necessary to comput cost here. But it is common to use cost 
        # in the termination condition of the loop
        cost = (error.T.dot(error))/(2*n)
        #stop iteration at convergence
        if cost <= 0.0001:
            break
        # your code above 
        # test
        # print(i, theta, cost)
        
    return theta
    raise NotImplementedError


# In[65]:


def kmeans_assignment(X, z):
    """
    Assign each instance to a cluster based on the shortest distance to all centroids. 
    Clusters are integers from 0 to K - 1.

    No loops allowed.
    
    Args:
        X: (n, d) NumPy array, each row is an instance of the data set
        z: (K, d) NumPy array, each row is the coordinate of a centroid.
        
    Returns:
        c: (n ,) NumPy array, the assignment of each instance to its closest centroid.
    """
    x2 = np.sum(X**2, axis=1, keepdims=True)
    y2 = np.sum(z**2, axis=1)
    xy = np.dot(X, z.T)

    # calculate l-2 distance
    dist = np.sqrt(x2 - 2*xy + y2)
    c = np.argmin(dist, axis=1)
    # minimum value of each raw of D
    return c
    
    # your code above 
    raise NotImplementedError


# In[66]:


def kmeans_update(X, c, K): 
    """
    Given the data set and cluster assignment, find the updated coordinates of all centroids.
    
    No loops allowed

    Args:
        X: (n, d) NumPy array, each row is an instance of the data set
        c: (n, ) NumPy array, the assignment of each instance to its closest centroid.
        K: scalar, the number of clusters
    Returns:
       z: (K, d) NumPy array, each row is the updated coordinates of a centroid.
    """
    
    # Your code below. (hint: use one-hot encoding)
    cluster_encoded_matrix = np.eye(K + 1)[c]
    centroid_matrix = np.dot(X.T, cluster_encoded_matrix) / np.sum(cluster_encoded_matrix, axis = 0)
    
    return centroid_matrix[:, 1:].T
    # your code above
    
    raise NotImplementedError


# In[ ]:





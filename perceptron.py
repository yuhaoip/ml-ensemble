# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    Perceptron classifier:
    
    parameters
    --------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the traning dataSet.
        
    Attributes
    -------------
    w_: 1d-array    such as: array([1,2,3,4,5])
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch.
    """
    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    # Get the weights
    def fit(self, X, y):
        """
        Fit training data:
        
        Parameters
        -------------
        X: {array-like}, shape=[n_samples, n_features]
        y: array-like, shape = [n_samples]
        
        Returns
        --------
        self: object
        """
        # initialize the weights
        self.w_ = np.zeros(1+X.shape[1])    # define x0=1
        self.errors_ = []                   # misclassification numbers
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                # update once misclassified
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    # ********************************************************
    """
    # Only used when applying to continual prediction, SSE cost func
    # And now batch gradient descend rather than random...
    """
    def fit_sse(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []          # Instead of misclassification nums, use cost func.
        
        for _ in range(self.n_iter):
            output = self.net_result(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
            
        return self

    # ********************************************************
    
    # After fitting, we have got the should-be weights!!
    # Get the predicted net results
    def net_result(self, X):
        """
        Calculate net input result
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    # Get the predicted classification
    def predict(self, X):
        """
        Return class label agter unit step
        """
        return np.where(self.net_result(X)>=0.0, 1, -1)
    
   

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    After fitting the classifier, we can print 
    the decision regions.
    """
    # setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    def show():
        # plot the regions
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
        fig.canvas.set_window_title('Decision region and learning rate curve')
        ax[0].contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        ax[0].set_xlim(xx1.min(), xx1.max())
        ax[0].set_ylim(xx2.min(), xx2.max())
        ax[0].set_title('decision region')
        
        # plot samples
        for idx, c1s in enumerate(np.unique(y)):
            ax[0].scatter(x=X[y==c1s,0], y=X[y==c1s,1], alpha=0.8,
                        c = cmap(idx), marker=markers[idx], label='class %d'%c1s)
            
        # plot the learning rate curve
        if 'cost_' in classifier.__dict__.keys():
            ax[1].plot(range(1,len(classifier.cost_)+1), classifier.cost_,
                       marker = 'o')
        else:
            ax[1].plot(range(1,len(classifier.errors_)+1), classifier.errors_,
                       marker = 'o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Sum-squared-error')
        ax[1].set_title('learning rate curve')
        
        plt.show()
            
    return show
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
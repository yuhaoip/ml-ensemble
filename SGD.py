# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class LinearClsf(object):
    """
    Adaptive linear classifier:
    
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
    cost_: list
        Number of misclassifications in every epoch.
    ------------
    """
    def __init__(self,eta=0.01,n_iter=10,random_state=None,shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False      # Just for clarity
        if random_state:
            np.random.seed(random_state)
            
            
    # To distinguish with np.random.shuffle(),
    # define _shuffle() as internal function, in fact
    # we do not need to import this function externally,
    # it justed be called and utilized internally.
    def _shuffle(self, X, y):
        """Shuffle trainning data"""
        # To coordinate xi vs. target, we need the same random indexes
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    # Just extract out this function!!
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1+m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):      # update via each training data
        """Apply learning rule to update the weights"""
        output = self.net_result(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error) # Stochastic Gradient Descend
        self.w_[0] += self.eta * error
        cost = (error**2).sum()/2.0
        return cost
    
    
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
        self._initialize_weights(X.shape[1])
        self.cost_ = []     
        
        for _ in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    
    def online_fit(self, X, y):
        """Fit training data without reinitializing the weights
        for online learning"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,y)
        return self
    
    
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
        ax[1].plot(range(1,len(classifier.cost_)+1), classifier.cost_,
                       marker = 'o')
        ax[1].set_ylabel('Sum-squared-error')
        ax[1].set_title('learning rate curve')
        
        plt.show()
            
    return show
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
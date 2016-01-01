# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisData(object):
    
    def __init__(self):
        iris = datasets.load_iris()
        X = iris.data[:,[2,3]]
        y = iris.target
        
        # Cross validation datas
        X_train,X_test,y_train,y_test = \
            train_test_split(X,y,test_size=0.3,random_state=1)
            
        # Normalization
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        
        # Recombine
        X_std_combined = np.vstack((X_train_std,X_test_std))
        y_combined = np.hstack((y_train, y_test))
        
        self.X_train = X_train
        self.X_train_std = X_train_std
        self.X_test = X_test
        self.X_test_std = X_test_std
        
        self.y_train = y_train
        self.y_test = y_test
        self.X_std_combined = X_std_combined
        self.y_combined = y_combined
    


def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):
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
    
    # plot the regions
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,4))
    fig.canvas.set_window_title('Decision region')
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_title('decision region')
    
    # plot samples
    # As you see, it also applied to multi-class
    for idx, c1s in enumerate(np.unique(y)):
        ax.scatter(X[y==c1s,0], X[y==c1s,1], alpha=0.8,
                    c = cmap(idx), marker=markers[idx], label='class %d'%c1s)
    
    # highlight test samples
    if test_idx:
        X_test = X[test_idx, :]
        ax.scatter(X_test[:,0], X_test[:,1], c='', alpha=1,
                    s=55, label='test set')
        
    plt.legend(loc='upper left')
        
        
        
        
        
        
        

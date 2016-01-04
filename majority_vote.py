# -*- coding:utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
# from sklearn.externals import six
import numpy as np


class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    """
    A majority vote ensemble classifier
    
    Parameters
    ------------
    classifiers: array-like, shape=[n_classifiers]
    Different classifiers for ensemble
    
    vote: str, {'classlable','probability'}
    Default: 'classlabel'
    If 'classlabel' the prediction is based on the
        argmax of class label. Else if 'probability',
        the argmax of the sum of probabilities is used
        to predict the class label.
    
    weights: array-like, shape=[n_classifiers]
    Optional, default: None
    If not None and provided weights list applied to 
        different classifiers.
    """
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    
    # Get fitted classifiers
    def fit(self, X, y):
        # Use LabelEncoder to ensure class labels start with 0,
        # which is import for np.argmax call in self.predict
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Returns
        ---------
        maj_vote: array-like, shape=[n_samples]
        Predicted class labels combining majority voting.
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.array([clf.predict(X) for clf
                                    in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax
                                           (np.bincount(x,weights=self.weights)),
                                           axis = 1, arr = predictions)
        maj_vote = self.labelenc_.inverse_transform(maj_vote)
            
        return maj_vote
    
    
    def predict_proba(self, X):
        """
        For example, we have 2 classifier and 3 training data, 2 class labels:
        predict_probas = [[[0.8,0.2],[0.7,0.3],[0.7,0.3]],[[0.6,0.4],[0.65,0.35],[0.8,0.2]]]
        probas = [[ [0.8,0.2],
                    [0.7,0.3],
                    [0.7,0.3]],
                    
                    [0.6,0.4],
                    [0.65,0.35],
                    [0.8,0.2] ]]
        avg_probas = ...
        """
        # Yes, we got an 3-D np.ndarray object.
        probas = np.asarray([clf.predict_proba(X) for clf
                             in self.classifiers_])
        avg_probas = np.average(probas, axis=0, weights=self.weights)
        return avg_probas
            
    
    #===========================================================================
    # def get_params(self, deep=True):
    #     """
    #     Get classifier parameter names for GridSearch.
    #     """
    #     if not deep:
    #         return super(MajorityVoteClassifier,
    #                      self).get_params(deep=False)
    #     else:
    #         out = self.named_classifiers.copy()
    #         for name, step in six.iteritems(self.named_classifiers):
    #             for key, value in six.iteritems(step.get_params(deep=True)):
    #                 out['%s__%s' % (name,key)] = value
    #         
    #         return out
    #===========================================================================
                                    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        














        
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
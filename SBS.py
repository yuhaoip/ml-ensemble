# -*- coding:utf-8 -*-
"""
Sequiential Backward Selection
具体来说，
1）初始化特征k=d，确定所要应用的estimator
2)当特征数目k仍高于k_features时：
    a) 计算所有k-1特征的estimator accuracy表现，将对应的
    accuracy和特征分别保存起来
    b）找出k-1维最高的accuracy及对应的特征
    c）将b）中的最高值另外保存起来，维度降低1
"""

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class SBS():
    def __init__(self,estimator,k_features,
                 scoring=accuracy_score,
                 test_size=0.25,random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=self.test_size,
                         random_state=self.random_state)
        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train,y_train,self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_,r=dim-1):
                score = self._calc_score(X_train,y_train,
                                         X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
    
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])
            dim -= 1
        # Finally, this is what k_score
        self.k_score_ = self.scores_[-1]
        return self
    
    
    def transform(self, X):
        return X[:, self.indices_]
    
      
    def _calc_score(self, X_train, y_train,
                    X_test, y_test, indices):
        self.estimator.fit(X_train[:,indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        
        return score
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
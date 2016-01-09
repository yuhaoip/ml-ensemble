# -*- coding:utf-8 -*-
"""
Get an intuitive comparison among Logistic
Regression, Decision Tree, KNN vs. Ensemble,
Random Forest and Adaboost based on iris datas.
Of course, we will plot them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from SBS import SBS
from sklearn.preprocessing import \
             StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cross_validation import \
             train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, \
                             AdaBoostClassifier

from sklearn.learning_curve import learning_curve,\
                                validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score,\
                            recall_score, f1_score
from sklearn.metrics.classification import accuracy_score




url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df = pd.read_csv(url, header=None)
# print 'The breast cancer-facor datasets magnitude: ', np.shape(df)
# print df.head()
X = df.iloc[:,2:].values
y = df.loc[:,1].values
X_train,X_test,y_train,y_test = \
            train_test_split(X,y,test_size=0.3,
                             random_state=1)
le = LabelEncoder()
y = le.fit_transform(y)


""""
--------------------
Logistic Regression:
--------------------
After plotting learning_curve and validation_curve,
I learn that the best tuned LR classifier should be:
lr = LogisticRegression(C=0.1, penalty='l2')

But Note: also, do need more data to fix slight overfitting issue.

And: Later I will plot all the classifiers' decision regions 
by using respective best params; Of course, just choose two of the
features but applying the same algorithm.
"""
def plot_data(X, y):
    # Plot the scatter figure to observe the datas
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
    ax[0].scatter(X[y==0,0],X[y==0,1],c='red',marker='o',alpha=0.5)
    ax[0].scatter(X[y==1,0],X[y==1,1],c='blue',marker='s',alpha=0.5)
    ax[0].set_title('Unscaled data')
    
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    ax[1].scatter(X_std[y==0,0],X_std[y==0,1],c='red',marker='o',alpha=0.5)
    ax[1].scatter(X_std[y==1,0],X_std[y==1,1],c='blue',marker='s',alpha=0.5)
    ax[1].set_title('Standardized data')
    
    mmc = MinMaxScaler()
    X_mmc = mmc.fit_transform(X)
    ax[2].scatter(X_mmc[y==0,0],X_mmc[y==0,1],c='red',marker='o',alpha=0.5)
    ax[2].scatter(X_mmc[y==1,0],X_mmc[y==1,1],c='blue',marker='s',alpha=0.5)
    ax[2].set_title('Nomalized data')
    plt.show()


## Train the LogisticRegression Classifier
# Diagnoise bias&variance by plotting Learning Curve,
# choose best param. C(tuning overfitting) by Validation Curve.
##Get the best param: {C=0.1}
def plot_learning_curve(param_C=0.1, validation_curve_=None):
    
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
        
        pipe_lr = Pipeline([('scl',StandardScaler()),
                            ('clf', LogisticRegression(
                        penalty='l2',C=param_C,random_state=1))])
        train_steps = np.linspace(0.1,1.0,10)
        train_sizes, train_scores, test_scores = \
                     learning_curve(estimator=pipe_lr,
                                    X=X_train,
                                    y=y_train,
                                    train_sizes=train_steps,
                                    cv=10)
        # Combine the 10 cv scores
        train_mean = np.mean(train_scores,axis=1)
        train_std = np.std(train_scores,axis=1)
        test_mean = np.mean(test_scores,axis=1)
        test_std = np.std(test_scores,axis=1)
        
        ax[0].plot(train_sizes, train_mean,
                   c='blue', marker='o', markersize=5,
                   label='training accuracy')
        ax[0].fill_between(train_sizes,
                         train_mean+train_std,
                         train_mean-train_std,
                         alpha=0.5, color='blue')
        ax[0].plot(train_sizes, test_mean,
                   c='green', linestyle='--',
                   marker='s',markersize=5,
                   label='validation accuracy')
        ax[0].fill_between(train_sizes,
                          test_mean+test_std,
                          test_mean-test_std,
                          alpha=0.15,color='green')
        ax[0].grid()
        ax[0].set_title('Learning curve with param C=%.3f' % param_C)
        ax[0].set_xlabel('Number of training samples')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend(loc='best')
        ax[0].set_ylim([0.8,1.1])
        
        # Validation_vurve is a good choice when there's only 
        # one param to tune. Such as inverse penality C.
        if not validation_curve_:
            ax[1].set_title('Validation curve not needed')
        else:
            param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
            train_scores_vd, test_scores_vd = validation_curve(
                                        estimator=pipe_lr,
                                        X=X_train,
                                        y=y_train,
                                        param_range=param_range,
                                        param_name='clf__C',
                                        cv=10)
            train_mean_vd = np.mean(train_scores_vd, axis=1)
            train_std_vd = np.std(train_scores_vd, axis=1)
            test_mean_vd = np.mean(test_scores_vd, axis=1)
            test_std_vd = np.std(test_scores_vd, axis=1)
            ax[1].plot(param_range, train_mean_vd, color='blue',
                       marker='o', markersize=5,
                       label='training accuracy')
            ax[1].fill_between(param_range, train_mean_vd+train_std_vd,
                             train_mean_vd-train_std_vd,
                             color='blue',alpha=0.15)
            ax[1].plot(param_range, test_mean_vd, c='green',
                       marker='s', markersize=5,
                       linestyle='--',
                       label='validation accuracy')
            ax[1].fill_between(param_range, test_mean_vd+test_std_vd,
                             test_mean_vd-test_std_vd,
                             color='green', alpha=0.15)
            ax[1].grid()
            ax[1].set_xscale('log')
            ax[1].set_ylim([0.8,1.0])
            ax[1].set_xlabel('Parameter C')
            ax[1].set_ylabel('Accuracy')
            plt.legend(loc='best')
            
        plt.show()


def get_best_lr(C=0.1,penalty='l2'):
    pipe_lr = Pipeline([('scl',StandardScaler()),
                    ('clf', LogisticRegression(
                    penalty='l2',C=C,random_state=1))])
    return pipe_lr


#########################################################
# 对于疾病监测这类问题，应该保证高的召回率
def get_f1_scores(clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_pred,y_pred=y_pred)
    
    return precision,recall,f1


# 传入estimator求得cross_validation分数  
def get_kf_scores(clf):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10)
    return scores


def get_scores(clf):
    validation_scores = get_kf_scores(clf)
    
    clf.fit(X_train,y_train)
    predict_scores = clf.score(X_train,y_train)
    test_scores = clf.score(X_test,y_test)
    
    print 'The validation scores: ', validation_scores
    print 'The means: ', np.mean(validation_scores)
    print 'The train scores: ', predict_scores
    print 'The test scores:', test_scores   
#########################################################    



"""
Above scores:
The validation mean: 0.9825
The train scores: 0.9899
The test scores:0.9649
---------------
Decision Tree
---------------
As see from above learning curves, LR has a good
generalization but still need more datas to figure
out slightly overfitting. Then does exists a better
classifier on the limited datasets?
I usually confused on traing decision tree as it's
easy to overfit especially when there's too much 
features. But how about 30 feats datasets.
"""     
# 决策树有很多参数，不过max_depth和max_leaf_nodes（不太容易控制）是互斥，
# 和min_samples_split(default=2)是依存的;下面，使用GridSearch方法
# 同时调节max_depth和min_samples_split来获取最优值
def get_best_tree():
    tree = DecisionTreeClassifier(random_state=1)
    max_depth = [3,4,5,6,7,8,9,10,11,20]
    min_samples_split = [2,3,4,5]
    params = [{'max_depth': max_depth,
               'min_samples_split': min_samples_split}]
    
    gs = GridSearchCV(estimator=tree,
                      param_grid=params,
                      scoring='accuracy',
                      cv=10)
    gs.fit(X_train, y_train)
    print 'The best params: ', gs.best_params_
    return gs.best_estimator_



"""
About result:
Get the best params:{max_depth:5, min_samples_split:4}
Validation scores:0.9552
Train scores:0.9975
Test scores: 0.9475
Badly, decision tree still easy to overfit.
Naturally, I think of Random Forest. And 
it's pretty easy, nearly no params to tune.
------------------------
Bagging: Random Forest
------------------------
Oops, train scores:1.000, test_scores:0.9532
!!More works need to be done!!
"""
def get_rf():
    rf = RandomForestClassifier(n_estimators=200,max_depth=None,
                                random_state=1,bootstrap=True)
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    y_train_score=accuracy_score(y_train,y_train_pred)
    y_test_score=accuracy_score(y_test,y_test_pred)
    
    print 'Random Forest train/test accuracies: %.4f/%.4f' \
            % (y_train_score,y_test_score)
    


"""
As gradient boosting classifier is very likely to
random forest, even better. Let't me have a try!
-------------
Gradient Boosting
-------------
Train score: 1.000
Test score:0.9415
"""
def get_ada():
    tree = DecisionTreeClassifier(max_depth=None)
    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=300,
                             learning_rate=0.1)
    ada.fit(X_train,y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    ada_train_score = accuracy_score(y_train, y_train_pred)
    ada_test_score = accuracy_score(y_test, y_test_pred)
    print 'AdaBoost train/test accuracies: %.4f/%.4f' \
            % (ada_train_score,ada_test_score)



"""
Gosh, I just wonder how will KNN behavior on such datasets?
For 2 reasons: 1)By GridSearch method, it's easy to control
the K value; 2)Maybe it exists some noise features, after
feature selection, based on illness detection, given suitable K 
value, I think it's supposed to have a good nieghbor voting charactic.
-------
KNN
-------
1)First, I tried KNN without feature selection.
best K = 2
Validation score:0.9145
Train score:0.9497
Test score: 0.8947; Obviously, severe overfitting.

2)Then, do feature selection by SBS method, which I have 
already done it.
Selected feature indices: [3,7,10,17,18,21,26]
Train score: 0.9623
Test score: 0.9474; Obviously, a huge enhencement!!
But............Seems still inferior to LR!!
"""
def get_best_knn():
    knn = KNeighborsClassifier(p=2,metric='minkowski')
    param_K = [2,3,4,5,6]
    gs = GridSearchCV(estimator=knn,
                      param_grid=[{'n_neighbors': param_K}],
                      scoring='accuracy',
                      cv=10)
    gs.fit(X_train,y_train)
    print 'The GridSearchScore: ', gs.grid_scores_
    print 'The best K value: ', gs.best_params_
    return gs.best_estimator_


def plot_kfeatures(return_clf=None):
    
    pipe_knn = Pipeline([('scl',StandardScaler()),
                         ('knn',KNeighborsClassifier(n_neighbors=2))])
    # 这里定义k_features=1是为了获取所有n个feature下的分类准确率
    sbs = SBS(estimator=pipe_knn,k_features=1)
    sbs.fit(X, y)
    
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat,sbs.scores_,marker='o')
    plt.ylim([0.5,1.0])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.title('KNN feature selection')
    plt.grid()
    plt.show()
    
    if return_clf:
        k7 = list(sbs.subsets_[-7])
        pipe_knn.fit(X_train[:,k7],y_train)
        print 'The selected feature indices: ', k7
        print 'Tranining accuracy: ',\
                pipe_knn.score(X_train[:,k7],y_train)
        print 'Testing accuracy: ', \
                pipe_knn.score(X_test[:,k7],y_test)







 

















        
        
        
        
        
        
        
        
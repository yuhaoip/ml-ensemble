# -*- coding:utf-8 -*-
"""
Adaboost算法：
(1)确定基本弱分类器;
(2)基本分类器训练过程：
        初始化训练数据集权值分布D1=(w11,w12,...,w1N),其中 w1i=1/N,(i=1,2,...,N);
        对于m=1,2,...,M:
    (a)根据Dm权值分布的数据集，求得使分类误差率em最低的基本分类器Gm(x)
                        其中：em = ∑ wmi*I(G(xi)!=yi), wmi=exp(-yi*f<m-1>(xi))
    (b)根据em计算分类器Gm(x)在最终分类器中的权重alpha<m>
            alpha<m> = 1/2log[(1-em)/em]
    (c)更新数据集权值分布
            Dm+1 = (w<m+1>1,w<m+1>2,...,w<m+1>N), 
            w<m+1>i = [wmi*exp(-alpha<m>*yi*Gm(xi))]/Zm
(3)构建基本分类器的线性组合：
    f(x)= ∑ alpha<m>*Gm(x)
        最终分类器G(x)=sign(f(x))
---------------------------
test dataSet:
dataMat = mat([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
labelMat = mat([1.0,1.0,-1.0,-1.0,1.0]).T
---------------------------
下面建立决策树桩的Adaboost算法
"""
from numpy import mat,shape,ones,zeros,log,exp,multiply,sign
from sklearn.datasets import load_iris


# 鹫尾花数据: 注意是多分类问题
def loadIris():
    iris = load_iris()
    irisDataArr = iris.data
    irisTargetArr = iris.target
    
    return mat(irisDataArr), mat(irisTargetArr).T


# 从文件加载数据
def loadData(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataArr = []; labelArr = []
    
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))    
            dataArr.append(lineArr)
            labelArr.append(float(curLine[-1]))
            
        return mat(dataArr), mat(labelArr).T
                   

## 构建单层决策树生成函数,包含三个循环（所有特征-->按步长遍布取值-->阈值比较时的正负类）
# 与特征阈值进行比较分类，结果按数据index存放在mX1数组中
def stumpClassify(dataMat,featureIndex,threadValue,info):
    predictedCsf = ones((shape(dataMat)[0],1))
    if info == 'lp':        # if large is positive category
        predictedCsf[dataMat[:,featureIndex]<threadValue] = -1.0
    else:
        predictedCsf[dataMat[:,featureIndex]>threadValue] = -1.0
    return predictedCsf

# 遍历特征与取值，找到权重误分率最低的决策树桩
def buildStump(dataMat,labelMat,D):
    m,n = shape(dataMat)
    minErr = float('inf')   # 循环外设定无穷大
    bestStump = {}          # 记录决策树桩的feature,threadValue以及error
    
    # 设定获取某一特征取值时的递进步长
    numSteps = 10.0; 
    for i in range(n):
        minValue=dataMat[:,i].min(); maxValue=dataMat[:,i].max()
        stepSize = (maxValue-minValue)/numSteps
        
        for j in range(-1,int(numSteps)+1):
            threadValue = minValue + j*stepSize
            
            for info in ['lp', 'sp']:
                predicted = stumpClassify(dataMat,i,threadValue,info)
                # 比较predicted和labelArr，确定哪些分类错误
                # 之后，错误的分类要乘以相应的权重，得到权重错误分类率
                # 所以错误分类最好在数组中表示出来
                errArr = ones((m,1))
                errArr[predicted==labelMat] = 0
                weightedErr = D.T * errArr
#                 # 打印信息，了解进展；用于调试，可注释掉
#                 print 'split featureIndex: %d, threadValue: %.2f, \
#                         info: %s, weightedError: %f' \
#                         % (i, threadValue, info, weightedErr)
                if weightedErr < minErr:
                    minErr = weightedErr
                    bestStump['bestFeatureIndex'] = i
                    bestStump['bestThreadValue'] = threadValue
                    bestStump['info'] = info
                    bestStump['minError'] = minErr
                    bestStump['bestCsf'] = predicted
                  
    return bestStump


                    
## 基于单层决策树的AdaBoost训练过程; 要将每个基分类器以适当方式保存起来！Such as dict!
## 如果新数据加入，则更新dataMat重新训练，得到新的weakCsfSet，用于预测新数据label
def adaBoostTraining(dataMat,labelMat,numIter=30):
    # 初始化权值分布
    m = shape(dataMat)[0]; D = mat(ones((m, 1))/m)
    # 存放每次迭代生成的弱分类器
    weakCsfSet = []
    # 初始化加法模型的预测结果
    addCsf = mat(zeros((m,1)))
    
    for _ in range(numIter):
        bestStump = buildStump(dataMat,labelMat,D)
#         print 'the weighted matrix D: ', D.T
        # 根据权重误分率，计算本次分类器的系数
        err = bestStump['minError']
        alpha = 0.5 * log((1-err)/err, 2)
        bestStump['alpha'] = alpha
        weakCsfSet.append(bestStump)
        
        # 更新下一次迭代的取值分布
        expon = exp(-alpha * multiply(bestStump['bestCsf'], labelMat))  # 列矩阵
        D = multiply(D, expon)
        D = D/D.sum()
        
        # 计算当前加法模型的误分类率，小于阈值(或为零)则退出
        addCsf += alpha * bestStump['bestCsf']    
#         print 'the added model estimation: ', addCsf
        addError = multiply(sign(addCsf)!=labelMat, ones((m,1)))
        addErrorRate = addError.sum()/m
#         print 'Currently, the additive model error rate: ', addErrorRate
        if addErrorRate == 0.0: break
        
    return weakCsfSet


## 将alpha，weakCsf分类结果从weakCsfSet中抽离出来
def adaBoostCsf(dataToClassify, weakCsfSet):
    dataMat = mat(dataToClassify)
    m = shape(dataMat)[0]
    addModelEst = mat(zeros((m,1)))
    
    for i in range(len(weakCsfSet)):
        modelEst = stumpClassify(dataMat, weakCsfSet[i]['bestFeatureIndex'], \
                                 weakCsfSet[i]['bestThreadValue'], \
                                 weakCsfSet[i]['info'])
        addModelEst += weakCsfSet[i]['alpha'] * modelEst
        
#         print addModelEst
    
    return sign(addModelEst)


## 在测试机上的错误率
def errorRateOnTest(estResult,testLabelMat):
    m = shape(testLabelMat)[0]
    errArr = mat(ones((m,1)))
    errRate = errArr[estResult!=testLabelMat].sum()/m
    return errRate
        




"""
*******************************************
下面使用scikit-learn库实现以上Adaboost过程...
*******************************************
"""
  
        
    
    
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

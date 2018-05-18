from hybrid import hmoea
from platypus import NSGAII,OMOPSO, Problem, Real,Binary
from platypus.types import Real, Binary, Permutation, Subset
import numpy as np
from sklearn.svm import SVC
from itertools import compress
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from skfeature.function.similarity_based import reliefF
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
import scipy.io
import os
import pandas as pd
import time
import xlsxwriter
datasets=[]
xlsxfile=[]
for k in os.listdir("dataset"):
    datasets.append(k)
    xlsxfile.append(str(k)[:-4]+'.xlsx')
#First function to optimize
def function2(x):
    roc_auc = dict()
    x=np.array(x).astype(int)
    print(x)
    if function1(x)>0:
        global feature_numbers
        feature_numbers1=np.array(list(compress(feature_numbers, x)))
        feature_numbers1=feature_numbers1.astype(int)
        feature_numbers1[:] = [x - 1 for x in feature_numbers1]
        X_new=X[:,feature_numbers1[:]]
        X_train, X_test, y_train, y_test = train_test_split(X_new, Y1)
        global model
        classifier = OneVsRestClassifier(model)
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    else :
        roc_auc["micro"]=0
    return 1-roc_auc["micro"]

#Second function to optimize
def function1(x):
    global feature_numbers
    feature_numbers1=np.array(list(compress(feature_numbers, x)))
    value = feature_numbers1.size
    return value

class Schaffer(Problem):

    def __init__(self):
        super(Schaffer, self).__init__(1, 2)
        global col
        self.types[:] = Binary(col)
    
    def evaluate(self, solution):
        x = solution.variables[:]
        print("x")
        solution.objectives[:] = [sum(x[0]), function2(x[0])]

start=time.time()
for data ,outfile in zip(datasets,xlsxfile):
    mat = scipy.io.loadmat('dataset/'+data)
    X = mat['data']
    Y = X[:, 0]
    Y=Y.astype(int)
    X=X[:,1:]
    
    score1 = reliefF.reliefF(X, Y)
    idx = reliefF.feature_ranking(score1)
    X=X[:,idx[0:60]]
    
    row,col=X.shape
    a=np.unique(Y)
    Y1 = label_binarize(Y, classes=a.tolist())
    n_classes = a.size
    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    n_samples, n_features = X.shape
    feature_numbers=np.linspace(1, len(X[0]), len(X[0]))        
    model = SVC(probability=True)
    algorithm=hmoea(Schaffer(),population_size = 15,nEXA=15)
        
    algorithm.run(generations=100)
    features_list=[]
    auc_score_list=[]
    for s in algorithm.result:
        features_list.append(s.objectives[0])
        auc_score_list.append(s.objectives[1])
    df = pd.DataFrame({"features selected":features_list  ,"auc score":auc_score_list})
    df.to_excel(writer, sheet_name='results')
    # plot the results using matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])

    plt.xlabel("no of features selected")
    plt.ylabel("1- AUC score")
    plt.savefig(str('Hybrid'+'_'+str(data)[:-4]))
    plt.gcf().clear()
    time=time.time()-start
    time_list=[]
    time_list.append(time)
    df = pd.DataFrame({"Time ":time_list})
    df.to_excel(writer, sheet_name='time')
    writer.save()


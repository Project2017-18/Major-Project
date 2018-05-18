from __future__ import print_function
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import os
import math
import random
from itertools import compress
from itertools import cycle
from sklearn import cross_validation
from scipy import stats
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import model_selection
#classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from skfeature.function.statistical_based import gini_index
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy import interp
import time
import xlsxwriter
from platypus import NSGAII,OMOPSO, Problem, Real,Binary

#model = KNeighborsClassifier()
model = SVC(probability=True)

datasets=[]
xlsxfile=[]
for k in os.listdir("dataset"):
    datasets.append(k)
    xlsxfile.append(str(k)[:-4]+'.xlsx')
    
solution=[]


def calacc(x):
    if function1(x)>0:
        global feature_numbers
        feature_numbers1=np.array(list(compress(feature_numbers, x)))
        feature_numbers1=feature_numbers1.astype(int)
        feature_numbers1[:] = [x - 1 for x in feature_numbers1]
        X_new=X[:,feature_numbers1[:]]
        X_train, X_test, y_train, y_test = train_test_split(X_new, Y)
        global model
        model=model.fit(X_train,y_train)
        pred=model.predict(X_test)
        acc=accuracy_score(y_test,pred)
        print(acc)
    else:
        acc=0
    return 1-acc

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

pop_size = 1000
max_gen = 10
start=time.time()
for data ,outfile in zip(datasets,xlsxfile):
    #getting the data
    accuracy_list=[]
    roc_list=[]
    genes_list=[]
    time_list=[]
    mat = scipy.io.loadmat('dataset/'+data)
    X = mat['data']
    
    Y = X[:, 0]
    Y=Y.astype(int)
    X=X[:,1:]
##    mRMR_sf,a,b=MRMR.mrmr(X,Y,n_selected_features=100)
##    X=X[:,mRMR_sf[0:100]]
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
    algorithm = OMOPSO(Schaffer(),None,swarm_size = 100,leader_size = 100)
    #algorithm = NSGAII(Schaffer(),population_size = 15)
    
    algorithm.run(2000)
    features_list=[]
    auc_score_list=[]
    for s in algorithm.result:
        features_list.append(s.objectives[0])
        auc_score_list.append(s.objectives[1])
    df = pd.DataFrame({"features selected":features_list  ,"auc score":auc_score_list})
    df.to_excel(writer, sheet_name='results')
    
    
    # plot the results using matplotlib
    import matplotlib.pyplot as plt

    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    
    plt.xlabel("no of features selected")
    plt.ylabel("1- AUC score")
    plt.savefig(str('MOPSO'+'_'+str(data)[:-4]))
    plt.gcf().clear()
time=time.time()-start
time_list=[]
time_list.append(time)
df = pd.DataFrame({"Time ":time_list})
df.to_excel(writer, sheet_name='time')
writer.save()

import pickle as pkl
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#   data preprocessing
data1 = pd.read_csv("new_data.csv")
data2 = pd.read_csv("test_set.csv")


data1 = np.array(data1)
label1 = data1[:, 4]
data1 =data1[:,5:]

data2 = np.array(data2)
label2 = data2[:, 4]
data2 =data2[:,5:]

train_set = data1
train_label = label1
test_set = data2
test_label = label2

#   SVM classifier
start =time.time()
cls_SVM = SVC(kernel='rbf',gamma='auto', C=0.9)
cls_SVM.fit(train_set, train_label)
pre_train = cls_SVM.predict(test_set)
ans_SVM = accuracy_score(test_label, pre_train)
end=time.time()
print('Running time: %s seconds'%(end-start))
print('SVM_accuracy: %s \n'%(ans_SVM))

# MLP classifier
start =time.time()
cls_MLP = MLPClassifier(hidden_layer_sizes=10 , max_iter=200)
cls_MLP.fit(train_set, train_label)
pre_train = cls_MLP.predict(test_set)
ans_MLP = accuracy_score(test_label, pre_train) 
end=time.time()
print('Running time: %s seconds'%(end-start))
print('MLP_accuracy: %s \n'%(ans_MLP))

 #   DT classifier
start =time.time()
cls_DT = tree.DecisionTreeClassifier(max_depth=20, criterion='entropy', splitter='best', min_samples_leaf=10)
cls_DT.fit(train_set, train_label)
pre_train = cls_DT.predict(test_set)
ans_DT = accuracy_score(test_label, pre_train)
end=time.time()
print('Running time: %s seconds'%(end-start))
print('DT_accuracy: %s \n'%(ans_DT))

#   KNN classifier
start =time.time()
cls_KNN = KNeighborsClassifier(n_neighbors=8, weights='uniform', algorithm='auto', leaf_size=30)
cls_KNN.fit(train_set, train_label)
pre_train = cls_KNN.predict(test_set)
ans_KNN = accuracy_score(test_label, pre_train)
end=time.time()
print('Running time: %s seconds'%(end-start))
print('KNN_accuracy: %s \n'%(ans_KNN))


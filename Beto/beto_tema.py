#import packages
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel
import re
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle as pkl
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, TimeDistributed, Dropout, LSTM
from keras.models import Sequential, load_model
from datasets import load_metric
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy import mean
from numpy import std
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import metrics

#Creamos las funciones pertinentes para cargar los datos
def clean_fake(text):
    cleaned_text_1 = re.sub('".*?"', '', text)
    cleaned_text = re.sub(r'\.(?=[^ \W\d])', '. ', cleaned_text_1)
    return cleaned_text

def clean(text):
    # removing all the characters other than alphabets
    cleaned_text_1= re.sub("[^a-zA-ZñÑ]", " ", text)
    cleaned_text_2 = re.sub(r'\W+', ' ', cleaned_text_1)
    # converting text to lower case
    cleaned_text = re.sub("\d+", " ", cleaned_text_2)
    #all lowercase
    cleaned_text= cleaned_text.lower()
    return cleaned_text

def normalize(s):  #Para quitar las tildes
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def load_big_dataset():
  direccion="C:/Users/pablo.garcia/Desktop/Fake news/Base de datos/BaseDatosFiltradaTRAIN.xlsx"
  fake_news= pd.read_excel(direccion)
  fake_news["Text"]= fake_news['Text'].apply(lambda x : normalize(x) )
  fake_news["Text"]= fake_news['Text'].apply(lambda x : clean_fake(x) )


  #clean text
  fake_news["clean_text"]= fake_news["Text"].apply(lambda x : clean(x) )


  #reset index
  fake_news = fake_news.reset_index(drop = True)

  return fake_news

def stop_words_fun(col):
  text = col 
  string = str(text)
  text_tokens = word_tokenize(string) #Separa las palabras y los símbolos
  tokens_without_sw = [word for word in text_tokens if not word in stop]
  joined = " ".join(tokens_without_sw)
  return joined

#load data and shuffle rows
df = load_big_dataset()
df= df.sample(frac=1).reset_index(drop=True)


#remove stopwords
stop = stopwords.words('spanish')
stop = set(stop)

def remove_stop_words(col):
  text = str(col)
  sent_text = nltk.sent_tokenize(text)
  return sent_text



df['stop_words'] = df['clean_text'].apply(lambda x : remove_stop_words(x))

df['clean_words'] = df['stop_words'].apply(lambda x: stop_words_fun(x)) #Elimino preposiciones, etc

df = df.drop_duplicates(subset = 'clean_words')

df['Topic'].value_counts()

def modelo_bert(nombre):
    tokenizer = AutoTokenizer.from_pretrained(nombre)
    model = AutoModel.from_pretrained(nombre)
    tokenized = df["clean_words"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, truncation = True, max_length = 512)))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [1]*(max_len-len(i)) for i in tokenized.values]) #Los ponemos todos de la misma longitud
    attention_mask = np.where(padded != 1, 1, 0)  #Ponemos un 0 en aquellos valores que se tuvieron que añadir para igualar longitudes
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
    return last_hidden_states
    
token=modelo_bert("dccuchile/bert-base-spanish-wwm-uncased")

features = token[0][:,0,:].numpy()
labels = df['Topic']

#Modelos de clasificación
np.random.seed(1234)
#Regresión logística
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"],'max_iter':[1,10,100,1000]}# l1 lasso l2 ridge
logreg=LogisticRegression(random_state=0)
logreg_cv=GridSearchCV(logreg,grid,cv=5)
logreg_cv.fit(features,labels)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


#Regresión Logística 
logreg = LogisticRegression(C=logreg_cv.best_params_["C"],penalty=logreg_cv.best_params_["penalty"])
cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
scores_RL = cross_val_score(logreg, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_RL), std(scores_RL)))

#Naive-Bayes
grid={'var_smoothing': np.logspace(0,-9, num=100)}# l1 lasso l2 ridge
gnb=GaussianNB()
gnb_cv=GridSearchCV(gnb,grid,scoring='accuracy',cv=5)
gnb_cv.fit(features,labels)

print("tuned hpyerparameters :(best parameters) ",gnb_cv.best_params_)
print("accuracy :",gnb_cv.best_score_)



gnb = GaussianNB(var_smoothing= gnb_cv.best_params_["var_smoothing"])

scores_GNB = cross_val_score(gnb, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_GNB), std(scores_GNB)))

#Random Forest 

param_grid = [
{'n_estimators': list(range(1,7)), 'max_features': list(range(1,15)), 
 'max_depth': [10, 20,30,40,50, None], 'bootstrap': [True, False]}
]
forest=RandomForestClassifier(random_state=0)
grid_search_forest = GridSearchCV(forest, param_grid, cv=5)
grid_search_forest.fit(features, labels)
#find the best model of grid search
grid_search_forest.best_estimator_
#bootstrap=False, max_depth=20, max_features=13,
                       #n_estimators=6, random_state=0
RF = RandomForestClassifier(bootstrap = grid_search_forest.best_params_["bootstrap"], max_depth=grid_search_forest.best_params_["max_depth"], max_features=grid_search_forest.best_params_["max_features"],
                       n_estimators=grid_search_forest.best_params_["n_estimators"], random_state=0)
cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
scores_RF = cross_val_score(RF, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_RF), std(scores_RF)))

#Neural Network
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
NN = MLPClassifier()
grid_search_nn = GridSearchCV(NN, param_grid, cv=5,refit=True,verbose=2)
grid_search_nn.fit(features, labels)
#find the best model of grid search
grid_search_nn.best_estimator_
NN = MLPClassifier(activation=grid_search_nn.best_params_["activation"], 
                   solver=grid_search_nn.best_params_["solver"],
                   hidden_layer_sizes=grid_search_nn.best_params_["hidden_layer_sizes"],
                   alpha=grid_search_nn.best_params_["alpha"],
                   learning_rate=grid_search_nn.best_params_["learning_rate"],
                   random_state=1)

scores_NN = cross_val_score(NN, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_NN), std(scores_NN)))

#K-NN

param_grid = [
{'n_neighbors': list(range(1,100))}
]
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid, cv=5)
grid_search_knn.fit(features, labels)
#find the best model of grid search
grid_search_knn.best_estimator_

knn = KNeighborsClassifier(n_neighbors=grid_search_knn.best_params_["n_neighbors"])
cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
scores_KNN = cross_val_score(knn, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_KNN), std(scores_KNN)))


#Árbol de decisión

param_grid = {"max_depth": range(1, 15),
             "min_samples_leaf": range(10, 50, 5),
             "min_samples_split": range(10, 50, 5),
             "criterion": ['gini', 'entropy']}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier(random_state=0)
grid_search_tree = GridSearchCV(estimator = dtree,
                          param_grid = param_grid,scoring='accuracy',
                          cv = n_folds)

# Fit the grid Search to the data
grid_search_tree.fit(features,labels)
print("best accuracy: ", grid_search_tree.best_score_)
print(grid_search_tree.best_estimator_)

tree = DecisionTreeClassifier(max_depth=grid_search_tree.best_params_["max_depth"], 
                              min_samples_leaf=grid_search_tree.best_params_["min_samples_leaf"],
                              min_samples_split=grid_search_tree.best_params_["min_samples_split"],
                              criterion=grid_search_tree.best_params_["criterion"],
                       random_state=0)
scores_DT = cross_val_score(tree, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_DT), std(scores_DT)))

#Bagging
param_grid = [
{'n_estimators': list(range(1,30))}
]

bagging = BaggingClassifier(random_state=0)
grid_search_bag = GridSearchCV(bagging, param_grid, cv=5)
grid_search_bag.fit(features,labels)
#find the best model of grid search
grid_search_bag.best_estimator_

bagging = BaggingClassifier(n_estimators=grid_search_bag.best_params_["n_estimators"])
scores_BAG = cross_val_score(bagging, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores_BAG), std(scores_BAG)))

#XGBoost
estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
parameters = {"booster": ['gbtree', 'gblinear','dart'],
              "eta":[0.3,0.6,1],
              "gamma":[25,50],
    'max_depth': [1,5,10],
    'n_estimators': [50,100]
}

grid_search_xgb = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,scoring='accuracy',
    verbose=True
)
grid_search_xgb.fit(features,labels)
print("best accuracy: ", grid_search_xgb.best_score_)
print(grid_search_xgb.best_estimator_)

xgb = XGBClassifier(base_score=0.5, booster=grid_search_xgb.best_params_["booster"], colsample_bylevel=None,
              colsample_bynode=None, colsample_bytree=None,
              enable_categorical=False, eta=grid_search_xgb.best_params_["eta"], gamma=grid_search_xgb.best_params_["gamma"], gpu_id=-1,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.5, max_delta_step=None, max_depth=grid_search_xgb.best_params_["max_depth"],
              min_child_weight=None,  monotone_constraints=None,
              n_estimators=grid_search_xgb.best_params_["n_estimators"], n_jobs=4, nthread=4, num_parallel_tree=None,
              predictor=None, random_state=42, reg_alpha=0, reg_lambda=0,
              scale_pos_weight=1, seed=42, subsample=None, tree_method=None,
              validate_parameters=1)

scores_XGB = cross_val_score(xgb, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores_XGB), std(scores_XGB)))

#SVM
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(features,labels)
print(grid.best_estimator_)


svc = SVC(C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
scores_SVM = cross_val_score(svc, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores_SVM), std(scores_SVM)))

#accuracy
print('Accuracy RL: %.3f (%.3f)' % (mean(scores_RL), std(scores_RL)))
print('Accuracy NB: %.3f (%.3f)' % (mean(scores_GNB), std(scores_GNB)))
print('Accuracy RF: %.3f (%.3f)' % (mean(scores_RF), std(scores_RF)))
print('Accuracy NN: %.3f (%.3f)' % (mean(scores_NN), std(scores_NN)))
print('Accuracy KNN: %.3f (%.3f)' % (mean(scores_KNN), std(scores_KNN)))
print('Accuracy Decision Tree: %.3f (%.3f)' % (mean(scores_DT), std(scores_DT)))
print('Accuracy Bagging: %.3f (%.3f)' % (mean(scores_BAG), std(scores_BAG)))
print('Accuracy XGB: %.3f (%.3f)' % (mean(scores_XGB), std(scores_XGB)))
print('Accuracy SVM: %.3f (%.3f)' % (mean(scores_SVM), std(scores_SVM)))
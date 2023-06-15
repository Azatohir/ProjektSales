# This is a sample Python script.
#uzyj xgbost, gridserach kilka razy sieci neuronowe kilka razy
#podaj wnioski

#Projekt podejscie numer 1

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import stats
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn import model_selection

samples = pd.read_csv("podstawy_sztucznej_inteligencji-main/ProjektDane/DataToWork.csv", sep=";", decimal=",")
print(samples.head())
print(samples.shape)
print(samples.columns)

# Delete nulls
null_counts = samples.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))

drop_list = ['Discount Band','Discounts'] #delete nulls
samplesWithoutNulls=samples.drop(drop_list, axis=1)
samplesWithoutNulls.head()

null_counts = samplesWithoutNulls.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))

drop_list = ['Segment', 'Country', 'Product', 'Date', 'Month Name'] #delete with words
samplesWithoutWords=samplesWithoutNulls.drop(drop_list, axis=1)
print(samplesWithoutWords.columns)

# #PCA and visualization
#
# df = pd.DataFrame(samplesWithoutWords)
# sns.pairplot(df)
# plt.show() #wizualizacja
#
# from sklearn.decomposition import PCA
# model = PCA()
#
# pca = PCA()
# pca.fit(samplesWithoutWords)
# print(pca.components_.shape)
# plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
# plt.show()
#
#
# from sklearn.preprocessing import StandardScaler
# X_std = StandardScaler().fit_transform(samplesWithoutWords)
#
# pca_features = model.fit_transform(X_std) # dwa wymiary, wiec rzutowanie jest na dwa wymiary
#
# xs = pca_features[:,0]
# ys = pca_features[:,1]
#
# plt.scatter(xs, ys)
# plt.axis('equal')
# plt.show()
#
# #wizualizacja na nowych wymiarach
# plt.figure(figsize=(15,10))
#
# plt.subplot(2,2,1)
# plt.scatter(xs, ys)
#
# plt.subplot(2,2,2)
# plt.scatter(np.zeros_like(ys),ys)
#
# plt.subplot(2,2,3)
# plt.scatter(xs, np.zeros_like(xs))
# plt.show()

#Grid
auto_target = samplesWithoutWords["Sales"]
auto_data = samplesWithoutWords.drop(["Sales"],axis=1)
auto_data.head()
auto_target.head()
y=auto_target
X=auto_data

# seed = 123
# kfold = model_selection.KFold(n_splits=5)
#
# grid_1 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), ElasticNet(alpha=1, random_state=seed)),
#                     param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
#                     'elasticnet__alpha': [0.01, 0.1, 1, 10]},
#                     cv=kfold,
#                     refit=True)
# grid_1.fit(X, y)
# print(grid_1.best_params_)
#
# grid_2 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), Lasso(alpha=1, random_state=seed)),
#                     param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
#                     'lasso__alpha': [0.01, 0.1, 1, 10]},
#                     cv=kfold,
#                     refit=True)
# grid_2.fit(X, y)
# print(grid_2.best_params_)
#
# grid_3 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1, random_state=seed)),
#                     param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
#                     'ridge__alpha': [0.01, 0.1, 1, 10]},
#                     cv=kfold,
#                     refit=True)
# grid_3.fit(X, y)
# print(grid_3.best_params_)
#
# grid_4 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression()),
#                     param_grid={'polynomialfeatures__degree': [1, 2, 3, 4]},
#                     cv=kfold,
#                     refit=True)
# grid_4.fit(X, y)
# print(grid_4.best_params_)
#
# grid_5 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), SVR()),
#                     param_grid={'polynomialfeatures__degree': [1, 2, 3, 4]},
#                     cv=kfold,
#                     refit=True)
# grid_5.fit(X, y)
# print(grid_5.best_params_)
#
# # grid_6 = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), RandomForestRegressor(random_state=seed)),
# #                       param_grid={'polynomialfeatures__degree': [1, 2, 3, 4]},
# #                       cv=kfold,
# #                       refit=True)
# # grid_6.fit(X, y)
# # print(grid_6.best_params_)
#
#
# models = []
# models.append(('ElasticNet', grid_1.best_estimator_))
# models.append(('Lasso', grid_2.best_estimator_))
# models.append(('Ridge', grid_3.best_estimator_))
# models.append(('LR', grid_4.best_estimator_))
# models.append(('SVR', grid_5.best_estimator_))
# #models.append(('RFR', grid_6.best_estimator_))
#
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
#
#
# train_data, test_data = train_test_split(samplesWithoutWords, test_size=0.2, random_state=42)
#
#
# X_test = test_data.drop(columns=["Sales"])
# y_test = test_data["Sales"]
#
# r2 = []
# explained_variance_score = []
# median_absolute_error = []
# mean_squared_error = []
# mean_absolute_error = []
# for name, model in models:
#     print(name)
#     print("R^2: {}".format(metrics.r2_score(y_test, model.predict(X_test)) ))
#     print("Explained variance score: {}".format( metrics.explained_variance_score(y_test, model.predict(X_test)) ))
#     print("Median absolute error: {}".format( metrics.median_absolute_error(y_test, model.predict(X_test)) ))
#     print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model.predict(X_test)) ))
#     print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y_test, model.predict(X_test)) ))
#     r2.append(metrics.r2_score(y_test, model.predict(X_test)))
#     explained_variance_score.append(metrics.explained_variance_score(y_test, model.predict(X_test)))
#     median_absolute_error.append( metrics.median_absolute_error(y_test, model.predict(X_test)))
#     mean_squared_error.append(metrics.mean_squared_error(y_test, model.predict(X_test)))
#     mean_absolute_error.append(metrics.mean_absolute_error(y_test, model.predict(X_test)))


#sa metryki i to
#teraz siec... omg ale super yey ekstra umieram z radosci
samplesTest = pd.read_csv("podstawy_sztucznej_inteligencji-main/ProjektDane/DataToWork.csv", sep=";", decimal=",")
samplesTrain = pd.read_csv("podstawy_sztucznej_inteligencji-main/ProjektDane/DataToWork.csv", skiprows=2,sep=";", decimal=",")

col=['Segment', 'Country', 'Product', 'Discount Band', 'Units Sold', 'Manufacture', 'Sale Price',
     'Gross Sale', 'Discounts', 'Sales', 'COGS', 'Profit', 'Date', 'Month_Number', 'Month Name', 'Year']
samplesTest.columns = col
samplesTrain.columns = samplesTest.columns

print(samplesTest.head())
print(samplesTrain.head())

test = samplesTest.drop(drop_list, axis=1)
train = samplesTrain.drop(drop_list, axis=1)

dataset = pd.concat([train,test])

dataset['Month_Number'] = dataset.Month_Number.replace({'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0,
                                                '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 1})

dataset.drop(["Year"],axis=1,inplace=True)


x = dataset.groupby('Manufacture')["Month_Number"].mean()

d = dict(pd.cut(x[x.index < 20],5,labels=range(5)))

dataset['Manufacture'] = dataset['Manufacture'].replace(d)

dataset = pd.get_dummies(dataset,drop_first=True)

train = dataset.iloc[:train.shape[0]]
test = dataset.iloc[train.shape[0]:]

X_train = train.drop("Month_Number",axis=1)
y_train = train.Month_Number

X_test = test.drop("Month_Number",axis=1)
y_test = test.Month_Number

from keras.models import Sequential
from keras.layers import Dense
import os
from keras.callbacks import History

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir

history = History()
model = Sequential()
model.add(Dense(100,activation="relu",input_shape=(X_train.shape[1],)))
model.add(Dense(50,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])
from keras.callbacks import TensorBoard

tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[tensorboard_cb])

#Potem Early stopping

from keras.callbacks import EarlyStopping
from sklearn import metrics
import numpy as np

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100, callbacks=[early_stopping])

# Evaluate the model
evaluation_result = model.evaluate(X_test, y_test)
print("Evaluation result:", evaluation_result)

# Predict probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to classes
y_pred = np.round(y_pred_prob).astype(int)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy:", accuracy)
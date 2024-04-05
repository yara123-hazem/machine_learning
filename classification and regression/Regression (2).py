import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# load the dataset
dataframe = read_csv('California_Houses.csv')  # read data
dataframe=dataframe.dropna()  # to removes the rows that contains NULL values.
y=dataframe["Median_House_Value"]   # save label in y
x=dataframe.drop(["Median_House_Value"], axis=1).astype('float64')   # save data in X
train_ratio = 0.70
valid_ratio = 0.15
test_ratio = 0.15
# split dataset into training set 70% and the remaining 30%
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=1 - train_ratio, random_state=42)
# then splitting the remaining X and Y into validation and dataset each 15%
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=test_ratio / (test_ratio + valid_ratio),
                                                        random_state=42)
# Linear Regression
linear_regr = LinearRegression()       # type of predictive modeling technique
linear_regr.fit(X_train, y_train)      # train the model on your data
score = linear_regr.score(X_test, y_test)   # evaluate the performance of a model on a test dataset.
print('Linear Regression Model:')
print('Score:', score)
y_pred = linear_regr.predict(X_test)    # used to make predictions on unseen data.
MSE = mean_squared_error(y_test, y_pred)
print('Mean Square Error:', MSE)
MAE = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:',MAE)


# lasso regression
alphas =[i for i in range(110,150)]   # range of alpha to select best alpha
scaler = StandardScaler().fit(X_train)  # standardize features by removing the mean and scaling to unit variance
X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)
X_valid=scaler.transform(X_valid)
error=np.empty(len(alphas))
for i in range (0,len(alphas)):  # loop with values of alpha to calculate the mean square error
    model = Lasso(alpha=alphas[i])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    error[i]= mean_squared_error(y_valid,y_pred)
index_best_alpha=error.argmin()   # the index of best alpha
best_alpha=alphas[index_best_alpha]   # best alpha
model = Lasso(alpha=best_alpha)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)   # calculate the mean square error of the best alpha
mae= mean_absolute_error(y_test,y_pred)   # calculate the mean_absolute_error of the best alpha
print("\nLasso Regression")
print("Best Alpha is ",best_alpha)
print('Mean Square Error of best alpha',mse)
print("Mean Absolute Error of best alpha:", mae)


# Ridge Regression
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)
params = {'alpha': (np.logspace(-8, 8, 100))}   # It will check from 1e-08 to 1e+08
ridge = Ridge()
# we used it for hyperparameter tuning to find the best# parameters for a given model.
ridge_model = GridSearchCV(ridge, params, cv = 10)
ridge_model.fit(X_train, y_train)
print('\nRidge Regression:')
print('Best Alpha:', ridge_model.best_params_.get('alpha'))    # best alpha
print('Best Score:', ridge_model.best_score_)  # the score of best alpha
ridge = Ridge(alpha=ridge_model.best_params_.get('alpha'))
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
mse =mean_squared_error(y_test, y_pred)   # mean_squared_error of the best alpha
print('Mean Square Error:', mse)
mae= mean_absolute_error(y_test, y_pred)      # mean absolute error of the best alpha
print("Mean Absolute Error:",mae)
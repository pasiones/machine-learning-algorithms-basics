from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
X_train = np.loadtxt('disease_X_train.txt')
X_test = np.loadtxt('disease_X_test.txt')
y_train = np.loadtxt('disease_y_train.txt')
y_test = np.loadtxt('disease_y_test.txt')

# Baseline
mean_train = y_train.mean()

baseline_predictions = [mean_train] * len(y_test)

baseline_mse = mean_squared_error(y_test, baseline_predictions)

print(f"Baseline MSE: {baseline_mse:.2f}")

# Linear model
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_linear_predict = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_linear_predict)

print(f"Linear regression MSE: {linear_mse:.2f}")

# Decision tree regressor
tree_model = DecisionTreeRegressor()

tree_model.fit(X_train, y_train)

y_tree_predict = tree_model.predict(X_test)

mse_tree = mean_squared_error(y_test, y_tree_predict)

print(f"Decision tree regressor MSE: {mse_tree:.2f}")

# Random forest regressor
forest_model = RandomForestRegressor()

forest_model.fit(X_train, y_train)

y_forest_predict = forest_model.predict(X_test)

mse_forest = mean_squared_error(y_test, y_forest_predict)

print(f"Random forest regressor MSE: {mse_forest:.2f}")


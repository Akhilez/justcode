from sklearn import linear_model
import matplotlib.pylab as plt
import numpy as np

# equation => y = 3x - 6

x_train = np.array([0, 1, 2, 3, 4, 5, 6])
y_train = np.array([-6, -3, 0, 3, 6, 9, 12])
x_test = np.array([-1, 3, 7, 14, -7, -14, 2.5])

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Predict Output
#predicted = linear.predict(x_test)
#print(predicted)

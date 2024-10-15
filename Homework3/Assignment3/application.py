import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.array(y).T
X = np.array(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE

alpha = [0.01, 0.1, 0.3, 0.5, 1.0]
epoch = [20, 30, 40, 50, 100]

best_alpha, best_epoch, best_train_loss = None, None, None
best_test_loss = float('inf')

for alpha in alpha:
    for n_epoch in epoch:
        w = train(Xtrain, Ytrain, alpha=alpha, n_epoch=n_epoch)

        #Make predictions
        y_train_pred = compute_yhat(Xtrain, w)
        y_test_pred = compute_yhat(Xtest, w)

        #Calculate  losses
        train_loss = compute_L(Ytrain, y_train_pred)
        test_loss = compute_L(Ytest, y_test_pred)


        print(f"Alpha: {alpha}, Epochs: {n_epoch}, Training Loss: {train_loss:.6f}, Testing Loss: {test_loss:.6f}")

        if test_loss < best_test_loss:
         best_alpha, best_epoch = alpha, n_epoch
         best_test_loss, best_train_loss = test_loss, train_loss

# Print the best results
print("\nBest Parameters:")
print(f"Best Alpha: {best_alpha}")
print(f"Best Epochs: {best_epoch}")
print(f"Best Training Loss: {best_train_loss:.6f}")
print(f"Best Testing Loss: {best_test_loss:.6f}")

#########################################


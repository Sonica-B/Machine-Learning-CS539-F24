from linear_regression import *
import numpy as np
import sys
'''
    Unit test:
    This file includes unit tests for linear_regression.py.
    You could test the correctness of your code by typing `pytest -v test.py` in the terminal.
'''


#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (70 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 


#-------------------------------------------------------------------------
def test_compute_Phi():
    ''' (10 points) compute_Phi'''
    x = np.array([1.,2.,3])

    Phi = compute_Phi(x,2) 
    assert np.allclose(Phi.T, [[1,1,1],[1,2,3]], atol = 1e-3) 

    Phi = compute_Phi(x,3) 
    assert np.allclose(Phi.T, [[1,1,1],[1,2,3],[1,4,9]], atol = 1e-3) 

    Phi = compute_Phi(x,4) 
    assert np.allclose(Phi.T, [[1,1,1],[1,2,3],[1,4,9],[1,8,27]], atol = 1e-3) 

#-------------------------------------------------------------------------
def test_compute_yhat():
    '''(10 points) compute_yhat'''

    # an example feature matrix with 2 features
    x = np.array([1.0, 2.0])
    w = np.array([0.5, -0.6])

    yhat = compute_yhat(x.T,w)
    assert np.allclose(yhat, -0.7, atol = 1e-3) 

    w = np.array([-0.5, 0.6])
    yhat = compute_yhat(x.T,w)
    assert np.allclose(yhat, .7, atol = 1e-3) 


	# an example feature matrix with 2 features for 3 instances
    w = np.array([0.5, -0.6])  # Create a 1D array
    x = np.array([[2.0, 0.8], [5.0, 0.9], [6.0, 3.0]])
    yhat = compute_yhat(x,w)

    assert np.allclose(yhat, [0.52, 1.96, 1.2], atol = 1e-3) 


	# an example feature matrix with 3 features for 2 instances
    w = np.array([0.7, 0.5, -0.1])
    x = np.array([[2.0, 0.8, 5.0], [0.9, 6.0, 3.0]])
    z = compute_yhat(x,w)
    assert np.allclose(z, [1.3,3.33], atol = 1e-3) 

    # an example feature matrix (4 instances, 2 features)
    Xtest = np.array([[0., 1.],
                  [1., 0.],
                  [2., 2.],
                  [1., 1.]])

    w = np.array([0.5, -0.6])

    # call the function
    Y= compute_yhat(Xtest, w )

    assert type(Y) == np.ndarray
    assert Y.shape == (4,) 

    Y_true = np.array([-0.6,  0.5, -0.2, -0.1])
    # check the correctness of the result 
    assert np.allclose(Y, Y_true, atol = 1e-2)


#-------------------------------------------------------------------------
def test_compute_L():
    '''(15 points) compute_L'''
    
    L= compute_L(np.array([1.,2.]), np.array([1.,2.]))
    assert np.allclose(L, 0., atol = 1e-3) 

    L= compute_L(np.array([1.,0.]), np.array([1.,2.]))

    assert np.allclose(L, 1., atol = 1e-3)

    L= compute_L(np.array([2.2,4.5]), np.array([2., 4.]))
    assert np.allclose(L, 0.0725, atol = 1e-3) 
    
    L= compute_L(np.array([0.1,0.5]), np.array([0.5, 0.5]))
    assert np.allclose(L, 0.04, atol = 1e-3) 


#-------------------------------------------------------------------------
def test_compute_dL_dw():
    '''(15 point) dL_dw'''
    y = np.array([0.5, 0.2, 1])
    yhat = np.array([0.56, 0.3, 0.95])
    Phi = np.array([[2.0, 0.8], [5.0, 0.9], [6.0, 3.0]])
    
    dL_dw = compute_dL_dw(y, yhat, Phi) 
    
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,) 

    dL_dw_true =np.array([0.10667, -0.004])
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)

#-------------------------------------------------------------------------
def test_update_w():
    '''(10 point) update_w'''
    w = np.array([0., 0.])
    dL_dw = np.array([1., 2.])

    w = update_w(w,dL_dw, alpha=.5) 
    
    w_true = - np.array([0.5, 1.])
    assert np.allclose(w, w_true, atol = 1e-3)

    w = update_w(w,dL_dw, alpha=1.) 
    w_true = - np.array([1.5, 3.])
    assert np.allclose(w, w_true, atol = 1e-3)

#-------------------------------------------------------------------------
def test_train():
    '''(10 point) train'''
    # an example feature matrix (4 instances, 2 features)
    Xtrain = np.array([[0., 1.],
                   [1., 0.],
                   [0., 0.],
                   [1., 1.]])

    Ytrain = np.array([0.1, 0.2, 0, 0.3])

    w = train(Xtrain, Ytrain, alpha=1., n_epoch = 20)
    assert np.allclose(Xtrain.dot(w), Ytrain, atol = 1e-3) 

    #------------------
    # another example
    Xtrain = np.array([[0., 1.],
                    [1., 0.],
                    [0., 0.],
                    [2., 0.],
                    [0., 2.],
                    [1., 2.]])
                   
    Ytrain = np.array([0.34, 0.38, 0, 0.77, 0.68, 1.07])
    w= train(Xtrain, Ytrain, alpha=0.1, n_epoch = 100)
    print(Xtrain.dot(w))
    assert np.allclose(Xtrain.dot(w), Ytrain, atol = 1e-2) 






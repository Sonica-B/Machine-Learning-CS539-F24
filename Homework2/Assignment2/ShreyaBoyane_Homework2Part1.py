import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
# However, importing the Python standard library is allowed: https://docs.python.org/3/library/
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 60 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.
'''

#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C = C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        counter = Counter(Y)
        total = len(Y)

        for val in counter.values():
            p = val / total
            e -= p * math.log(p,2)  #Entroy

        #########################################
        return e
    

            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        unique, counts = np.unique(X, return_counts=True)
        total = len(X)
        ce = 0
        for val, count in zip(unique, counts):
            subset = Y[X == val] # Sample filtering (X==val)
            ce += (count / total) * Tree.entropy(subset) # Weighted entropy of each partition

        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    

        g = (Tree.entropy(Y)) - (Tree.conditional_entropy(Y,X))
 
        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        best_g = -float('inf')
        i = 0
        for index in range(X.shape[0]):  # Iterate through each attribute
            gain = Tree.information_gain(Y,X[index,:])
            if gain > best_g:
                best_g = gain
                i = index


   

 
        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        Xunique = np.unique(X[i])
        #Yunique = np.unique(Y[:,i])
        C={}

        for val in Xunique:
            index = np.where(X[i] == val)[0]
            print(index)
            Xsub = X[:, index]
            Ysub = Y[index]
            C[val] = Node(Xsub,Ysub)



        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        s = None
        if len(np.unique(Y)) == 1:
            s = True
        else:
            s = False
        
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE



        for i in range (X.shape[0]):
            if len(np.unique(X[i])) > 1:
                return False




 
        #########################################
        return True
    
            
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        y = Counter(Y).most_common(1)[0][0]  # Returns the most frequent label

 
        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape p by n.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # Stopping condition 1: all labels are the same
        if Tree.stop1(t.Y):
            t.isleaf = True
            t.p = t.Y[0]  # Set the node prediction to the common label
            return t

        # Stopping condition 2: all attribute values are the same
        if Tree.stop2(t.X):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)  # Set to the most common label
            return t

        # Find the best attribute to split
        best_attr = Tree.best_attribute(t.X, t.Y)
        t.i = best_attr
        t.p = Tree.most_common(t.Y)

        if best_attr is None:
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            return t

        # Initialize children dictionary
        t.C = Tree.split(t.X, t.Y, best_attr)


        # Recursively build the tree for each child node
        for val, child in t.C.items():
           # Tree.build_tree(child)
            t.C[val] = Tree.build_tree(child)
        return t
   


 
        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Tree.build_tree(Node(X, Y))

 
        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vector of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        if t.isleaf:
            y= t.p
        else:
            val = x[t.i]
            if val in t.C:
                return Tree.inference(t.C[val], x)
            #else:
            #   return None  # If the value doesn't exist in the tree

        y = t.p

        return y

        #########################################

    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        Y = np.array([Tree.inference(t,X[:,instance]) for instance in range(X.shape[1])])

        #########################################
        return Y



    #--------------------------


def load_dataset(filename = 'data1.csv'):

    '''
        Load dataset 1 from the CSV file: 'data1.csv'.
        The first row of the file is the header (including the names of the attributes)
        In the remaining rows, each row represents one data instance.
        The first column of the file is the label to be predicted.
        In remaining columns, each column represents an attribute.
        Input:
            filename: the filename of the dataset, a string.
        Output:
            X: the feature matrix, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the dataset, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

#     try:
    data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
    #print(data)

   # print(data.shape)
    #Separate the features (X) and labels (Y)
    X = data[1:, 1:].transpose()  # All rows, all columns except the first (features)
    #print(X.shape)
    Y = data[1:, 0]  # All rows, only the first column (labels)
    #print(X)
    #print("Dataset loaded successfully.")
    return X, Y

X, Y = (load_dataset())
Tree.split(X,Y,1)

    #########################################


"""

   This is a code of a Artificial Neural Network (ANN), written from scratch. 
   The input data (training, validation and test datasets) were extracted from 
   the sklearn.datasets.make_moons. 


                                      Costa-Duarte, M. V. - 19/06/2018
"""
import numpy as np 
import sklearn 
import sklearn.datasets 
import h5py
import matplotlib.pyplot as plt

def sigmoid(x, deriv = False):
    """
     Sigmoid activation function

    """
    if deriv == False:
        return 1.0/(1+ np.exp(-x))
    else:
        return x * (1.0 - x)

def softmax(A):
    """
      Softmax activation function

    """
    exp_A = np.exp(A)
    return exp_A / exp_A.sum(axis=1, keepdims=True)

def tanh(A, deriv = False):
    """
     tanh activation function

    """
    if deriv == False:
        return np.tanh(A)
    else:
        return (1.0 - np.power(np.tanh(A), 2))

def load_dataset():
    """
       Loading dataset and define Training and Test samples
       
    """

    # Loading training dataset    
    hf = h5py.File('training.hdf5', 'r')
    X_train = hf['X'].value
    Y_train = hf['Y'].value
    hf.close()

    # Loading validation dataset    
    hf = h5py.File('validation.hdf5', 'r')
    X_validation = hf['X'].value
    Y_validation = hf['Y'].value
    hf.close()

    # Loading test dataset    
    hf = h5py.File('test.hdf5', 'r')
    X_test = hf['X'].value
    Y_test = hf['Y'].value
    hf.close()

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test
 
def predict(model, X): 
    """
      Given the trained model, this routine predicts the classification of elements. 

    """

    probs, a1 = foward_propagation(model, X)

    return np.argmax(probs, axis=1) 
 
def build_train_model(ann_model): 
    """
      This routine starts and Ws and biases arrays and train the Neural Networks
      using the Training sample and checking its performance on Validation dataset.

    """

    # Initialize the weights (random values) and bias (=0) 

    W1 = np.random.randn(ann_model.n_input_dim, ann_model.n_hlayer) / np.sqrt(ann_model.n_input_dim) 
    b1 = np.zeros((1, ann_model.n_hlayer)) 
    W2 = np.random.randn(ann_model.n_hlayer, ann_model.n_output_dim) / np.sqrt(ann_model.n_hlayer) 
    b2 = np.zeros((1, ann_model.n_output_dim)) 
 
    # Define model which will contains Ws and biases

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 

    # Loop over the n_passes... 
    for i in range(0, ann_model.n_passes): 
 
        # Forward propagation 

        probs, a2 = foward_propagation(model, ann_model.X_train)

        # Backpropagation 

        model = back_propagation(ann_model, probs, a2, model)

        if i % 100 == 0: 
          print("Score after iteration %d: %f" %(i, score(predict(model, ann_model.X_validation), ann_model.Y_validation))) 

    return model

def foward_propagation(model, X):
    """
      Foward propagation
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 

    a1 = X.copy()
    z1 = a1.dot(W1) + b1 
    a2 = tanh(z1, deriv = False) # hidden layer activation function: sigmoid
    z2 = a2.dot(W2) + b2 
    probs = softmax(z2)             # output layer activation function: softmax

    return probs, a2

def back_propagation(ann_model, a3, a2, model):
    """
      Back Propagation
    """

    # Loading model

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 

    # Backpropagating..

    # Define delta2
    delta2 = a3 
    delta2[range(ann_model.n_train), ann_model.Y_train] -= 1 
    delta1 = (delta2).dot(W2.T) * tanh(a2, deriv = True)

    # Weights

    dW2 = a2.T.dot(delta2)
    dW1 = ann_model.X_train.T.dot(delta1)

    # Bias

    db2 = (delta2).sum(axis=0)
    db1 = (delta1).sum(axis=0)

    # Add regularization terms (b1 and b2 don't have regularization terms) 

    dW2 += ann_model.reg_lambda * W2 
    dW1 += ann_model.reg_lambda * W1

    # Update parameter (gradient descen)

    W1 += -ann_model.epsilon * dW1 
    b1 += -ann_model.epsilon * db1 
    W2 += -ann_model.epsilon * dW2 
    b2 += -ann_model.epsilon * db2 

    # Update parameters to the model 

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 

    return model

class ann:
    """
       This class keeps all ANN parameters and datasets

    """
    def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):

        # ANN parameter
        self.epsilon = 0.001                # Learning rate
        self.reg_lambda = 0.00             # Regularization term
        self.n_hlayer = 10                 # Hidden layer
        self.n_input_dim = np.shape(X_train)[1]
        self.n_passes = 10000
        self.n_output_dim = 2             # Output

        # Training 
        self.X_train = X_train 
        self.Y_train = Y_train
        self.n_train = len(X_train[:, 0])

        # Validation
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.n_validation = len(X_validation[:, 0])

        # Test
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_test = len(X_test[:, 0])

def score(class_out, Y):
    """
      Calculate the Score -> rate of correctly classified objects.
    """

    return sum([1. for i in range(len(Y)) if Y[i] == class_out[i]]) / float(len(Y))

#######################################################################################
 
if __name__ == '__main__':

    print('Loading dataset..')

    # Loading datasets 

    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = load_dataset()

    # Put everything in a class

    ann_model = ann(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)

    print("Build and Train ANN model..")

    model = build_train_model(ann_model) 

    score_final = score(predict(model, X_test), Y_test)     
    print("Final Score: %f" % score_final)

    exit()
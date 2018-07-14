import numpy as np
import sklearn          
import sklearn.datasets 
import matplotlib.pyplot as plt
import matplotlib.pyplot

# Initialize weights and biases
def init_weights_biases(input_dim, hdim, output_dim):
	# Initialize the parameters to random values. We need to learn these.
	np.random.seed(0)
	W1 = np.random.randn(input_dim, hdim) 
	b1 = np.zeros((1, hdim))
	W2 = np.random.randn(hdim, output_dim) 
	b2 = np.zeros((1, output_dim))
	return W1,b1,W2,b2

# Softmax Activation
def softmax(input):
	exp_scores = np.exp(input)
	return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

# Cross Entropy Loss 
def log_likelihood(model, y):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	z1 = np.dot(X, W1) + b1
	# RELU
	# a1 = np.maximum(0, z1)
	# tanh
	a1 = np.tanh(z1)  
	z2 = np.dot(a1, W2) + b2
	probs = softmax(z2)

	# error = -1/len(probs)*np.sum(y*np.log(probs)+(1-y)*np.log(1-probs))
	error = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(error)

	data_loss += reg_strength/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	# return np.sum(np.nan_to_num(-y*np.log(probs)-(1-y)*np.log(1-probs)))
	return 1./num_examples * data_loss

# def delta(probs, y):
# 	return (probs-y)

# Predict 
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)
    return np.argmax(probs, axis=1)# returns max of prob vector and puts out a single class value for x

def plot_decision_boundary(pred_func):
    # Set min and max values depending on data matrix and give it some padding-->(1)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them, that will be the points we put the contour on-->(2)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid, that gives the contour the color-->(3)
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)# makes contour plots of z, surface is xx and yy-->(4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)#plots the normal scatter plot -->(5)
    

X, y = sklearn.datasets.make_moons(500, noise=0.20)

num_examples = len(X) # training set size
input_dim = 2         # input layer dimensionality
hdim = 4              # hidden layers 
output_dim = 2        # output layer dimensionality

num_passes = 10000

# Gradient descent parameters 
lr_rate = 0.00001      # learning rate for gradient descent
reg_strength = 0.0 # regularization strength

# init weights and biases
W1, b1, W2, b2 = init_weights_biases(input_dim, hdim, output_dim)

# Gradient descent. For each iteration
for i in range(0, num_passes):
	# Forward Propagation
	z1 = np.dot(X, W1) + b1
	a1 = np.tanh(z1)
	z2 = np.dot(a1, W2) + b2
	probs = softmax(z2)

	# Backward Propagation
	delta3 = probs
	delta3[range(num_examples), y] -= 1
	dW2 = np.dot(a1.T, delta3)
	db2 = np.sum(delta3, axis=0, keepdims=True)

	# for RELU
	# delta2 = np.dot(delta3, W2.T)
	# delta2[a1 <=0 ] = 0
	delta2 = np.dot(delta3, W2.T) * (1-np.power(a1, 2))
	dW1 = np.dot(X.T, delta2)
	db1 = np.sum(delta2, axis=0)

	# Add regularization 
	dW1 += reg_strength * W1
	dW2 += reg_strength * W2

	# Gradient descent parameter update
	# update parameters with learning rate lr_rate
	W1 += lr_rate * W1
	b1 += lr_rate * b1
	W2 += lr_rate * W2
	b2 += lr_rate * b2

	model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

	if i % 1000 == 0:
          print ("Loss after iteration %i: %f" %(i, log_likelihood(model, y))) 

plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()
















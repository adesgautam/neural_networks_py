
import numpy as np
import sklearn          
import sklearn.datasets 
import matplotlib.pyplot as plt
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist


class NN():
	# Initialize weights and biases
	def __init__(self, input_dim, hdim1, hdim2, output_dim):
		# Initialize the parameters to random values. We need to learn these.
		np.random.seed(0)
		self.W1 = np.random.uniform(-1, 1, size=(input_dim, hdim1))
		self.b1 = np.zeros((1, hdim1))
		self.W2 = np.random.uniform(-1, 1, size=(hdim1, hdim2))
		self.b2 = np.zeros((1, hdim2))
		self.W3 = np.random.uniform(-1, 1, size=(hdim2, output_dim))
		self.b3 = np.zeros((1, output_dim))
		# print(self.W1.shape, self.W2.shape, self.W3.shape)

	# Softmax Activation
	def softmax(self, input):
		exp_scores = np.exp(input) 
		return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

	def forward_pass(self, X, y):
		# W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
		z1 = np.dot(X, self.W1) + self.b1
		# RELU
		a1 = np.maximum(0, z1)
		# tanh
		# a1 = np.tanh(z1)
		z2 = np.dot(a1, self.W2) + self.b2
		# RELU
		a2 = np.maximum(0, z2)
		# tanh
		# a2 = np.tanh(z2)
		z3 = np.dot(a2, self.W3) + self.b3
		probs = self.softmax(z3)
		return probs

	def accuracy(self, X, y):
		probs = self.forward_pass(X, y)
		correct = [1 if i[0]>0 else 0 for i in probs]
		correct = [1 if a == b else 0 for (a, b) in zip(np.array(correct), y)]
		accuracy = np.sum(correct) / len(correct)
		return accuracy*100

	def predict(self, X):
		z1 = np.dot(X, self.W1) + self.b1
		a1 = np.maximum(0, z1) # RELU
		# tanh a1 = np.tanh(z1)
		z2 = np.dot(a1, self.W2) + self.b2
		a2 = np.maximum(0, z2) # RELU
		# tanh a2 = np.tanh(z2)
		z3 = np.dot(a2, self.W3) + self.b3
		probs = self.softmax(z3)
		prediction = np.argmax(probs, axis=1)
		return prediction

	# Cross Entropy Loss 
	def loss(self, X, y, reg_strength):
		# print(X.shape, self.W1.shape, self.b1.shape)
		z1 = np.dot(X, self.W1) + self.b1
		# RELU
		a1 = np.maximum(0, z1)
		# tanh
		# a1 = np.tanh(z1)
		z2 = np.dot(a1, self.W2) + self.b2
		# RELU
		a2 = np.maximum(0, z2)
		# tanh
		# a2 = np.tanh(z2)
		z3 = np.dot(a2, self.W3) + self.b3
		# Softmax function
		probs = self.softmax(z3)
		# print(probs)
		N = y.shape[0]
		
		# error = np.log(probs).T*y+np.log(1-probs).T*(1-y)
		error = -np.log(probs[range(N), y])
		# error = np.sum(np.nan_to_num(-y*np.log(probs)-(1-y)*n 
		
		data_loss = np.sum(error) / N
		reg_loss = reg_strength/2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
		loss = data_loss + reg_loss

		# Backward Propagation
		delta4 = probs
		delta4[range(N), y] -= 1  # loss (delta3-y) 
		delta4 /= N  # normalize (input_samples X output_size)
		
		dW3 = np.dot(a2.T, delta4) # (hdim2 X output_size)
		db3 = np.sum(delta4, axis=0)
		
		delta3 = np.dot(delta4, self.W3.T)
		delta3[a2 <=0 ] = 0 # for RELU
		# delta3 = np.dot(delta3, W3.T) * (1-np.power(a2, 2)) # for tanh
		dW2 = np.dot(a1.T, delta3)
		db2 = np.sum(delta3, axis=0)
		
		delta2 = np.dot(delta3, self.W2.T)
		delta2[a1 <=0 ] = 0 # for RELU
		# delta2 = np.dot(delta2, W2.T) * (1-np.power(a1, 2)) # for tanh
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)

		# Add regularization 
		dW3 += reg_strength * self.W3
		dW2 += reg_strength * self.W2
		dW1 += reg_strength * self.W1
		
		grads = {'W1':dW1, 'b1':db1, 'W2':dW2, 'b2':db2, 'W3':dW3, 'b3':db3}
		return loss, grads
	
	def fit(self, X, y, X_val, y_val, epochs=10, lr_rate=0.01, reg_strength=1e-6, batch_size=32, verbose=False):
		loss_history, train_acc_history, val_acc_history = [], [], []

		num_train = X.shape[0]
		iterations_per_epoch = (num_train // batch_size)+1
		
		# Gradient descent. For each batch
		for epoch in range(1, epochs+1):

			batch_start = 0
			batch_end = batch_size
			train_acc_epoch = 0
			val_acc_epoch = 0

			for i in range(0, iterations_per_epoch):
				# sample_indices = np.random.choice(np.arange(num_train), batch_size)
				
				if num_train-batch_start < batch_size:
					X_batch = X[batch_start:num_train+1]
					y_batch = y[batch_start:num_train+1]
				else:
					X_batch = X[batch_start:batch_end]
					y_batch = y[batch_start:batch_end]

				loss, grads = self.loss(X_batch, y_batch, reg_strength)
				loss_history.append(loss)
				
				# Gradient descent parameter update
				# update parameters with learning rate lr_rate
				self.W1 += -lr_rate * grads['W1']
				self.b1 += -lr_rate * grads['b1']
				self.W2 += -lr_rate * grads['W2']
				self.b2 += -lr_rate * grads['b2']
				self.W3 += -lr_rate * grads['W3']
				self.b3 += -lr_rate * grads['b3']
				
				# lr_rate *= lr_decay
				batch_start += batch_size
				batch_end += batch_size

				# Calculating accuracy 
				train_acc_batch = (self.predict(X_batch) == y_batch).mean()
				val_acc_batch = (self.predict(X_val) == y_val).mean()

				train_acc_epoch += train_acc_batch
				val_acc_epoch += val_acc_batch

			if verbose:
				train_acc = train_acc_epoch/iterations_per_epoch
				val_acc = val_acc_epoch/iterations_per_epoch
				train_acc_history.append(train_acc_epoch/iterations_per_epoch)
				val_acc_history.append(val_acc_epoch/iterations_per_epoch)
				print('Iteration {0} / {1}:  Loss {2}\tTrain accuracy: {3}\tValidation accuracy: {4}'.format(epoch, epochs, round(loss,4), round(train_acc,2), round(val_acc,2)))

		return loss_history, train_acc_history, val_acc_history
	
# START
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

input_dim = 28*28     # input layer dimensionality
hdim1 = 20            # hidden layer1 
hdim2 = 30            # hidden layer2 
output_dim = 10        # output layer dimensionality
epochs = 20
lr_rate = 0.1       # learning rate for gradient descent
# lr_decay = 0.001
reg_strength = 0.0    # regularization strength
batch_size = 128
verbose = True

nn = NN(input_dim, hdim1, hdim2, output_dim)
loss_history, train_acc, test_acc = nn.fit(X_train, y_train, X_test, y_test, epochs, lr_rate, reg_strength, batch_size, verbose)

# plot the loss history
# plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# plt.subplot(1,2,2)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy history')
plt.show()

predict = True
if predict:
	for _ in range(1,10):
		i = np.random.randint(1, X_test.shape[0])
		prediction = nn.predict(X_test[i])
		print(prediction)
		plt.imshow(X_test[i].reshape(28,28))
		plt.show()

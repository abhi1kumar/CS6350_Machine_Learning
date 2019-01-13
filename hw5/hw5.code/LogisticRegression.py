import numpy as np
from common_math import sign
from common_math import sigmoid

################################################################################
# class Logistic Regression
################################################################################
class LogisticRegression(object):
	def __init__(self, dim = 0, sigmaSq = 1):
		self.dim      = dim + 1            # #dimensions of perceptron + bias
		self.w        = np.zeros(self.dim) # weights
		self.sigmaSq  = sigmaSq            # Regularisation Hyperparameter

	def get_weight(self):
		return self.w

	def init_random(self):
		self.w = 0.02*np.random.rand(self.dim) - 0.01 # small number between -0.01 and 0.01
	
	def init_zeros(self):
		self.w = np.zeros(self.w.shape)
	
	def update(self, lr, x, y):
		# Update assumes single example is passed
		pred     = self.dot_with_weight(x)
		sig_pred = sigmoid(-y*pred)		
		self.w   = (1 - 2*lr/self.sigmaSq) * self.w + lr * sig_pred * y * np.append(x,1)
			
	def predict(self, x):
		if(x.ndim == 1):	
			return sign(sigmoid(np.array([self.w.dot(np.append(x,1))]))-0.5)
		else:
			return sign(sigmoid(np.append(x,np.ones([len(x), 1]),1).dot(self.w.T))-0.5)

	# This is required in updation step
	def dot_with_weight(self, x):
		if(x.ndim == 1):	
			return np.array([self.w.dot(np.append(x,1))])
		else:
			return np.append(x,np.ones([len(x), 1]),1).dot(self.w.T)

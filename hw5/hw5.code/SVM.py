import numpy as np
from common_math import sign

################################################################################
# class SVM
################################################################################
class SVM(object):
	def __init__(self, dim = 0, C = 0):
		self.dim      = dim + 1            # #dimensions of perceptron + bias
		self.w        = np.zeros(self.dim) # weights
		self.C        = C                  # Regularisation Hyperparameter

	def get_weight(self):
		return self.w

	def init_random(self):
		self.w = 0.02*np.random.rand(self.dim) - 0.01 # small number between -0.01 and 0.01
	
	def init_zeros(self):
		self.w = np.zeros(self.w.shape)
	
	def update(self, lr, x, y):
		pred = self.dot_with_weight(x)
		
		# Update assumes single example is passed		
		if (pred[0]*y <= 1):	
			self.w = (1 - lr) * self.w + lr * self.C * y * np.append(x,1)
		else:
			self.w = (1 - lr) * self.w
	
	def predict(self, x):
		if(x.ndim == 1):	
			return sign(np.array([self.w.dot(np.append(x,1))]))
		else:
			return sign(np.append(x,np.ones([len(x), 1]),1).dot(self.w.T))

	# This is required in updation step
	def dot_with_weight(self, x):
		if(x.ndim == 1):	
			return np.array([self.w.dot(np.append(x,1))])
		else:
			return np.append(x,np.ones([len(x), 1]),1).dot(self.w.T)

import numpy as np
from signum import sign

################################################################################
# class Perceptron
################################################################################
class Perceptron(object):
	def __init__(self, dim = 0, avg_flag = False):
		self.dim      = dim + 1            # #dimensions of perceptron + bias
		self.w        = np.zeros(self.dim) # weights
		self.w_avg    = np.zeros(self.dim) # averaged weights
		self.avg_flag = avg_flag		   # This flag counts the avg flag
		self.cnt      = 0				   # Counter for number of examples

	def get_weight(self):
		return self.w


	def init_random(self):
		self.w = 0.02*np.random.rand(self.dim) - 0.01 # small number between -0.01 and 0.01
		self.w_avg = self.w.copy()


	def update(self, lr, x, y):
		self.w = self.w + lr * y * np.append(x,1)

	
	def update_avg(self):
		self.cnt += 1
		if(self.avg_flag == True):
			alpha = float(1)/self.cnt
			self.w_avg = (1-alpha)*self.w_avg + alpha*self.w


	def predict(self, x):
		if(self.avg_flag == False):
			if(x.ndim == 1):	
				return sign(np.array([self.w.dot(np.append(x,1))]))
			else:
				return sign(np.append(x,np.ones([len(x), 1]),1).dot(self.w.T))
		else:
			if(x.ndim == 1):	
				return sign(np.array([self.w_avg.dot(np.append(x,1))]))
			else:
				return sign(np.append(x,np.ones([len(x), 1]),1).dot(self.w_avg.T))


	def predict_train(self,x):
		if(x.ndim == 1):	
			return sign(np.array([self.w.dot(np.append(x,1))]))
		else:
			return sign(np.append(x,np.ones([len(x), 1]),1).dot(self.w.T))

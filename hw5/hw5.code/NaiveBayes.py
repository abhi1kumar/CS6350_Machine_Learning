import numpy as np
from common_math import sign
from common_math import sigmoid

################################################################################
# class Naive Bayes
################################################################################
class NaiveBayes(object):
	def __init__(self, dim = 0, smoothing = 1, num_values_for_each_feature = 2, num_classes = 2):
		self.dim                         = dim                          # dimensions or number of features
		self.num_classes                 = num_classes
		self.num_values_for_each_feature = num_values_for_each_feature		
		self.smoothing                   = smoothing                    # Smoothing Hyperparameter
		
		self.prior_cnt                   = np.zeros(self.num_classes,dtype=int)
		self.likelihood_cnt              = np.zeros((self.num_classes, self.dim, self.num_values_for_each_feature),dtype=int)
		self.example_cnt                 = 0
		self.prior						 = self.prior_cnt.copy()
		if(self.smoothing > 0):
			self.likelihood				 = (1/self.num_values_for_each_feature)*np.ones((self.num_classes, self.dim, self.num_values_for_each_feature))
		else:
			self.likelihood				 = np.zeros((self.num_classes, self.dim, self.num_values_for_each_feature))
		self.eps						 = 1e-8

	# Calculates probabilities
	# https://stackoverflow.com/a/23944658
	def get_probability(self, fav_cases, sample_space, smooth_flag = True):
		if (isinstance(sample_space, np.ndarray)):
			if (sample_space.shape[0] > 1):
				sample_space = 	sample_space.reshape((-1,1))

		if (smooth_flag):
			return (fav_cases + self.smoothing)/(sample_space + self.num_values_for_each_feature*self.smoothing)
		else:
			return (fav_cases)/(self.eps + sample_space)

	
	def update(self, x, y):
		# Update assumes single example is passed
		if (y == -1):
			class_index = 0
		else:
			class_index = 1

		# Update counts of prior
		self.example_cnt            += 1
		self.prior_cnt[class_index] += 1

		# Update probability of prior
		self.prior     = self.get_probability(self.prior_cnt, self.example_cnt, False)

		# Update the likelihood counts first and then update its matrix	
		# Vectorised implementation is order of magnitudes faster than the 
		# non-vectorised implementation	
		index = np.round(x).astype(int)
		self.likelihood_cnt[class_index, range(self.dim), index] += 1		
		self.likelihood    [class_index] = self.get_probability(self.likelihood_cnt[class_index], np.sum(self.likelihood_cnt[class_index],axis=1), smooth_flag = True)


	def predict(self, x):
		if(x.ndim == 1):	
			x = np.array([x])
		
		feature_index = np.round(x).astype(int)
		out = np.zeros(x.shape[0],dtype=int)

		for i in range(x.shape[0]):
			# Multidimensional indexing taken from
			# https://scipy-cookbook.readthedocs.io/items/Indexing.html#Multidimensional-list-of-locations-indexing
			temp = self.likelihood[:,range(self.dim),feature_index[i]]							
			
			# Append  priors to the temp
			temp = np.hstack((temp,self.prior.reshape((-1,1))))	
			
			# Use logarithms to add and select with maximum value
			temp = np.argmax(np.sum( np.log(temp), axis=1))
			
			if(np.abs(temp) < self.eps):
				temp = -1
			out[i] = temp
		return out

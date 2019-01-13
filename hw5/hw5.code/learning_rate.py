import numpy as np

################################################################################
# Decays a learning rate based on iteration number t
################################################################################
def decay_lr(base_lr, t):
	return float(base_lr)/(1+t)

################################################################################
# Returns a learning rate based on optimisation problem for Aggressive Perceptron
################################################################################
def aggressive_lr(mu, x, y, w):
	return float(mu - y * w.dot(np.append(x,1)))/(1+x.dot(x)+1)

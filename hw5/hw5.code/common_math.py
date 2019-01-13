import numpy as np
import math

################################################################################
# Signum function implementation
# Returns 1 when input >= 0 and -1 otherwise
################################################################################
def sign(input):
	output = input.copy()
	
	output[output>=0] = 1
	output[output< 0] = -1
	
	return output


################################################################################
# Sigmoid function implementation
# Numerically stable implementation from https://stackoverflow.com/a/25164452
################################################################################
def sigmoid(input):
	output = input.copy()

	z = np.exp(-output[output >= 0])
	output[output >= 0] = 1 / (1 + z)

	z = np.exp(output[output <   0])	
	output[output <  0] = z / (1 + z)

	return output

import numpy as np
################################################################################
# Returns 1 when input >= 0 and -1 otherwise
################################################################################
def sign(input):
	output = input.copy()
	
	output[output>=0] = 1
	output[output< 0] = -1
	
	return output

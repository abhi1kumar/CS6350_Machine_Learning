import numpy as np
import math
from random import randint

################################################################################
# Loads a data file into numpy array
################################################################################
def load_file(path, max_col_prior=0):
	matrix = []
	label  = []

	max_col_cnt = 0

	with open(path) as f:
		for line in f:
			data  = line.split()
			label.append(float(data[0])) # label value
			row   = []

			col_cnt = 0
			for i, (idx, value) in enumerate([item.split(':') for item in data[1:]]):
				
				# Use difference in expected missing thing to get the indexes
				# which are zeroed
				n = int(idx) - (i + 1) 

				# for all missing entries				
				for _ in range(n):
					row.append(0)
					col_cnt += 1 
					
				row.append(float(value))
				col_cnt += 1
			matrix.append(row)
			
			if(col_cnt > max_col_cnt):
				max_col_cnt = col_cnt		

	#Check which of the two max_col one from prior or one in this set is maximum
	if(max_col_cnt < max_col_prior):
		max_col_cnt = max_col_prior

	#print("max col count = " + str(max_col_cnt))

	# Append zeros to columns which have less column count initially
	for i in range(len(matrix)):
		for j in range(max_col_cnt - len(matrix[i])):
			matrix[i].append(0)

	return np.array(matrix), np.array(label), max_col_cnt#.astype(int)

################################################################################
# Gets accuracy from ground truth and predictions
################################################################################
def get_accuracy(ground, predicted):
	correct = 0
	if (ground.shape[0] != predicted.shape[0]):
		print("Array sizes do not match")
		return 0.0
	
	correct = np.sum(ground == predicted)
	return float(correct)*100/ground.shape[0]

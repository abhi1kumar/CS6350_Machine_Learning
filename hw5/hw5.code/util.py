import numpy as np
import math
from random import randint
from sklearn.metrics import confusion_matrix

################################################################################
# Loads a data file into numpy array
#
# Bug fixed. Previous indices should be taken into account.
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
			prev_idx = 0
			for i, (idx, value) in enumerate([item.split(':') for item in data[1:]]):
				
				curr_idx = int(idx) - 1
				# Use difference in expected missing thing to get the indexes
				# which are zeroed
				if i == 0:					
					n = curr_idx
				else:
					n = curr_idx - (prev_idx+1) 
				prev_idx = curr_idx

				#print("Current ids = " + str(curr_idx))
				#print("Diff = " + str(n))

				# for all missing entries				
				for _ in range(n):
					row.append(0)
					col_cnt += 1
	
				row.append(float(value)) 
				#print(row)
				#print("\n")
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


################################################################################
# Gets Precision, Recall and F-scores ground truth and predictions
################################################################################
def get_f_scores(ground, predicted):
	correct = 0
	if (ground.shape[0] != predicted.shape[0]):
		print("Array sizes do not match")
		return np.zeros((3,))
	
	mat = confusion_matrix(ground, predicted)
	eps = 1e-8
	#print(mat)

	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
	# It will be increasing order of labels
	#	
	#   --           --
	#  | TN    |    FP |
	#  |---------------|
	#  | FN    |    TP |
	#   --           --
	p = float(mat[1][1])/(mat[0][1] + mat[1][1] + eps)
	r = float(mat[1][1])/(mat[1][0] + mat[1][1] + eps)

	f = 2*p*r/(p + r + eps)

	return np.array([p, r, f])


import util
from data import Data
from tree import Tree
from tree import Node

import numpy as np
import math
from random import randint


################################################################################
# Q3 (1)
################################################################################
DATA_DIR = 'data/'

# Test data
data2     = np.loadtxt(DATA_DIR + 'test.csv', delimiter=',', dtype = str)
data_obj2 = Data(data = data2)

# Train data
data      = np.loadtxt(DATA_DIR + 'train.csv', delimiter=',', dtype = str)
data_obj  = Data(data = data)

print("Filling Missing Entries\n...")
# Fill the missing entries in the data
majority  = util.get_majority_column_data(data_obj)
data_obj  = util.fill_data(data_obj ,majority,data )
data_obj2 = util.fill_data(data_obj2,majority,data2)
print("...\n...\n...\nFilled Missing Entries\n")

myTree = Tree()
util.ID3(data_obj, data_obj.attributes, myTree.get_root(), myTree, False, 2)

# print("--------------- Printing Tree --------------------")
myTree.print_tree(myTree.get_root(),0)

# Accuracy Prediction
acc = util.prediction_accuracy(data_obj , myTree)
acc = util.prediction_accuracy(data_obj2, myTree)

# Depth Prediction
print("\nDepth of Tree = " + str(myTree.get_depth(myTree.get_root())))

################################################################################
# Q3 (2a)
################################################################################
DATA_DIR = 'data/CVfolds_new/'

num_folds = 5
max_depth = [1,2,3,4,5,10,15]

data = []
acc = np.zeros((len(max_depth),num_folds))

for i in range(num_folds):
	data.append(np.loadtxt(DATA_DIR + 'fold' + str(i+1) + '.csv', delimiter=',', dtype = str))

for i in range(len(max_depth)):
	print("Depth = " + str(max_depth[i]))
	for j in range(num_folds):

		if(j==0):
			start = 1			
			train_data = data[1]
			test_data  = data[0]
		else:
			start = 0
			train_data = data[0]
			test_data  = data[j]
	
		# Train data
		for k in range(start+1,num_folds):
			if(k != j):		
				train_data = np.concatenate([train_data, data[k]] ,axis=0)

		train_data_obj = Data(data = train_data)

		# Test data
		test_data_obj  = Data(data = test_data)

		#print("Filling Missing Entries\n...")
		# Fill the missing entries in the data
		majority       = util.get_majority_column_data(train_data_obj)
		train_data_obj = util.fill_data(train_data_obj, majority, train_data)
		test_data_obj2 = util.fill_data(test_data_obj , majority, test_data)
		#print("...\nFilled Missing Entries\n")		

		myTree = Tree()
		util.ID3(train_data_obj, train_data_obj.attributes, myTree.get_root(), myTree, True, max_depth[i])

		acc[i][j] = util.prediction_accuracy(test_data_obj, myTree)

# Calculate mean and standard deviations	
m = np.mean(acc,axis=1)
s = np.std (acc,axis=1)

for i in range(len(max_depth)):
	print("Depth = " + str(max_depth[i]) + ". Mean accuracy = " + str(m[i]) + ". Std deviation = " + str(s[i]))


################################################################################
# Q3 (2b)
################################################################################
DATA_DIR = 'data/'

# Train data
data      = np.loadtxt(DATA_DIR + 'train.csv', delimiter=',', dtype = str)
data_obj  = Data(data = data)

# Test data
data2     = np.loadtxt(DATA_DIR + 'test.csv', delimiter=',', dtype = str)
data_obj2 = Data(data = data2)

# Fill the missing entries in the data
majority  = util.get_majority_column_data(data_obj)
data_obj  = util.fill_data(data_obj ,majority,data )
data_obj2 = util.fill_data(data_obj2,majority,data2)

myTree = Tree()
util.ID3(data_obj, data_obj.attributes, myTree.get_root(), myTree, True, 5)
acc    = util.prediction_accuracy(data_obj2, myTree)

from data import Data
from tree import Tree
from tree import Node

import numpy as np
import math
from random import randint

################################################################################
# Takes an array of probabilities and turns into entropy
################################################################################
def entropy(p):
	ent = 0
	for i in range(len(p)):
		ent = ent - p[i]*math.log(p[i],2) #entropy is to the base 2
	return ent


################################################################################
# Calculates information of data based on an attribute or the whole data
################################################################################
def info(data_obj,attribute = None):
	info = 0

	# if no attribute is specified, get distribution of whole data
	if attribute is not None:	
		# get how many unique values of that attribute are there.
		uniq_vals = data_obj.attributes[attribute].possible_vals
		num_uniq_vals = len(uniq_vals)
	else:
		num_uniq_vals = 1

	num_points = len(data_obj.get_column('label'))

	for i in range(num_uniq_vals):
		if attribute is not None:
			subset = data_obj.get_row_subset(attribute,uniq_vals[i])
		else:
			subset = data_obj
		
		num_points_subset = len(subset)
		wt_subset = float(num_points_subset)/num_points
		
		# get counts of individual labels
		subset_labels = subset.get_column('label')		
		bins, counts = np.unique(subset_labels, return_counts=True)		

		# get probabilities	
		p = np.zeros(len(bins),)		
		for j in range(len(bins)):
			p[j] = float(counts[j])/num_points_subset 
	
		ent = entropy(p)
		info = info + wt_subset*ent
	return info


################################################################################
# Information gain based on an attribute 
################################################################################
def info_gain(data_obj,attribute = None):
	info_gain = info(data_obj) - info(data_obj,attribute)
	return info_gain


################################################################################
# Returns a new dictionary after making a copy of the original dictionary and 
# then removing that key
################################################################################
def remove_key(d,key):
	d_new = d.copy()
	del d_new[key]
	return d_new


################################################################################
# Implementation of Decision Tree using ID3 algorithm
#
# Inputs:
# data_obj    = data object for which we have to run ID3
# attributes  = a dictionary which tells which features to look for
# parent      = parent of the root node
# tree        = tree object
# limit_depth = a boolean value which says whether we have to limit depth or not
# max_depth   = 
################################################################################
def ID3(data_obj, attributes, parent, tree, limit_depth = False, max_depth = 10):
	label = data_obj.get_column('label')

	if(len(label)<=0):	
		# No data		
		return

	elif(len(np.unique(label))==1):
		# Object of only 1 label in the tree
		# Add a node with labels
		n = Node(np.unique(label)[0],True)		
		tree.add_node(n, parent)

	elif(len(attributes)<=0):
		# Add majority label to the tree as the node		
		# first get counts of individual labels	
		bins, counts = np.unique(label, return_counts=True)
		n = Node(bins[np.argmax(counts)],True)
		tree.add_node(n, parent)
			
	else:
		if(limit_depth):
			if(max_depth < 0):
				print("Max-depth should be greater than 0. Aborting!!!")
				return

		# Information gain for each features
		info_gain_per_feature = {}
		for key in attributes:
			info_gain_per_feature[key] = info_gain(data_obj,key)
			# print(key + "," + str(info_gain_per_feature[key]))
		
		# Choose the best feature and the possible values
		best_feature        = max(info_gain_per_feature, key=info_gain_per_feature.get)
		best_feature_values = data_obj.attributes[best_feature].possible_vals

		# Add a node
		n = Node(best_feature, False)

		# Add all possible directions in which node can go
		for i in range(len(best_feature_values)):
			# partition into subset based on different values
			data_subset_obj = data_obj.get_row_subset(best_feature, best_feature_values[i])
			
			# if non-zero items in the subset data
			if(data_subset_obj.raw_data.shape[0] > 0):
				n.add_value(best_feature_values[i])
		
		tree.add_node(n, parent)

		# Check depth of the tree after adding this node.
		if(limit_depth):
			depth = tree.get_depth(tree.get_root())

			if(depth > max_depth):
				# Donot grow the tree instead add label nodes				
				tree.del_node(n, parent)

				# Add majority label to the tree as the node		
				# first get counts of individual labels	
				bins, counts = np.unique(label, return_counts=True)
				n = Node(bins[np.argmax(counts)],True)
				tree.add_node(n, parent)
				return

		# pop this feature from dictionary
		attributes_new = remove_key(attributes,best_feature)
		
		for i in range(len(n.value)):			
			# partition into subsets based on different values
			data_subset_obj = data_obj.get_row_subset(best_feature, n.value[i])
			
			if(parent is None):			
				new_parent = tree.get_root()
			else:
				new_parent = parent.child[-1]			
		
			ID3(data_subset_obj, attributes_new, new_parent, tree, limit_depth, max_depth)


################################################################################
# Prediction using Decision Tree
#
# Inputs:
# root		 = node from which looking started 
# tree       = tree object
# data		 = data_row
# attributes = dictionary of attributes
################################################################################
def predict(root, tree, data, attributes):
	if(root is not None):
		
		# label is the leaf node
		if(len(root.child) == 0):
			return root.feature
		else:
			node_attribute = root.feature
			node_attr_ind  = attributes[node_attribute].index
			data_val       = data[node_attr_ind+1] #1st one is label
			
			if(data_val in root.value):
				child_index = root.value.index(data_val)
			else:
				#print("No node direction matches the value of attribute")				
				child_index = randint(0, len(root.value)-1)

			label = predict(root.child[child_index], tree, data, attributes)
			return label	
	else:
		return


################################################################################
# Prediction Accuracy using Decision Tree
#
# Inputs:
# data_obj   = data_obj
# tree       = tree object
################################################################################
def prediction_accuracy(data_obj, tree):
	test_data = data_obj.raw_data
	num_data_points = test_data.shape[0]
	#print("\nComputing Accuracy on " + str(num_data_points) + " dataset")
	
	correct = 0
	for i in range(num_data_points):
		predicted = predict(tree.get_root(), tree, test_data[i], data_obj.attributes)
		ground    = test_data[i][0]
		if (predicted == ground):
			correct = correct + 1
		#if (i%1000 == 0):
		#	print(str(i) + "\tPoints processed. Accuracy = " + str(100*float(correct)/(i+1)))
	
	accuracy = 100*float(correct)/num_data_points
	#print("-------------------------------------------------------------------")
	print("Final Accuracy ovr " + str(num_data_points) + " datapoints = " + str(accuracy))
	#print("Error (in %)  over " + str(num_data_points) + " datapoints = " + str(100-accuracy))
	#print("-------------------------------------------------------------------")

	return accuracy


################################################################################
# Prediction Label using Decision Tree
#
# Inputs:
# data_obj   = data_obj
# tree       = tree object
################################################################################
def prediction_label(data_obj, tree):
	test_data = data_obj.raw_data
	num_data_points = test_data.shape[0]

	output = np.zeros(num_data_points)	

	for i in range(num_data_points):
		predicted = predict(tree.get_root(), tree, test_data[i], data_obj.attributes)
		output[i] = predicted

	return output


################################################################################
# Gets majority entry of each column in train data
################################################################################
def get_majority_column_data(data_obj):
	majority = {}

	for key in data_obj.attributes:
		# print (key)
		column_data = data_obj.get_column([key]).flatten();
		
		values,counts = np.unique(column_data,return_counts=True)
		majority[key] = values[np.argmax(counts)]
		
	return majority


################################################################################
# Fills missing entry by majority values of the train data
################################################################################
def fill_data(data_obj, majority, data):
	missing_data = '?'
	
	if data_obj is None:
		return None

	for key in data_obj.attributes:		
		index = data_obj.attributes[key].index
		
		for i in range(data.shape[0]-1): #1st one is label
			# if missing data found
			if(data[i][index+1] == missing_data):
				data[i][index+1] = majority[key] #1st one is label

	# Replace data_obj with the new data
	data_obj= Data(data = data)

	return data_obj	



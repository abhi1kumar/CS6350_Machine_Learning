import util
import learning_rate
from common_math        import sign
from common_math        import sigmoid
from SVM                import SVM
from LogisticRegression import LogisticRegression
from NaiveBayes         import NaiveBayes

import tree_util
from data import Data
from tree import Tree
from tree import Node

import numpy as np
import math
from   random import randint
#import matplotlib.pyplot as plt

epoch_cv   = 20
epoch_test = 20
limit_min  = 50 
limit_max  = 80

np.random.seed(231)
################################################################################
# Load CV Data
################################################################################
print("\nReading data ...")
DATA_DIR = 'data/CVSplits/'
num_folds = 5

data_cv = []
label_cv = []
max_col_prior = 0

# First get what is the maximum number of features across all folds
for i in range(num_folds):
	_, _, max_col_prior = util.load_file(DATA_DIR + 'training0' + str(i) + '.data', max_col_prior)
	
#print(max_col_prior)

for i in range(num_folds):
	data_fold, label_fold, max_col_prior = util.load_file(DATA_DIR + 'training0' + str(i) + '.data', max_col_prior)
	data_cv.append (data_fold)
	label_cv.append(label_fold)


################################################################################
# Load Train and Test Data
################################################################################
DATA_DIR = 'data/'
data_tr, label_tr, max_col_prior = util.load_file(DATA_DIR + 'train.liblinear', max_col_prior)
data_te, label_te, max_col_prior = util.load_file(DATA_DIR + 'test.liblinear' , max_col_prior)

print(max_col_prior)

"""
################################################################################
# Majority Baseline
################################################################################
print("\n************************************************************************")
print("*************************** Majority Label *****************************")
print("************************************************************************")

(values,counts)      = np.unique(label_tr,return_counts=True)
majority_label       = values[np.argmax(counts)]

prediction      = np.ones(label_te.shape)*majority_label
print ("Accuracy for majority label on test set = %.2f" %(util.get_accuracy(label_te, prediction)))
"""

################################################################################
# Prepare validation splits
################################################################################
print("\nPreparing cross-validation splits ...")
train_split_data  = []
train_split_label = []
test_split_data   = []
test_split_label  = []

# For each fold
for j in range(num_folds):

	if(j==0):
		start = 1			
		train_data = data_cv[1]
		train_label = label_cv[1]

		test_data  = data_cv[0]
		test_label = label_cv[1]
	else:
		start = 0
		train_data = data_cv[0]
		train_label = label_cv[0]

		test_data  = data_cv[j]
		test_label = label_cv[j]

	# Train data and label
	for k in range(start+1,num_folds):
		if(k != j):		
			train_data  = np.concatenate([train_data,  data_cv[k]] , axis=0)
			train_label = np.concatenate([train_label, label_cv[k]], axis=0)

	train_split_data .append(train_data)
	train_split_label.append(train_label)

	test_split_data .append(test_data)
	test_split_label.append(test_label)


################################################################################
# SVM with decaying learning rate
################################################################################
print("\n************************************************************************")
print("******************** SVM (Decaying learning rate) **********************")
print("************************************************************************")

lr_list = [    1, 0.1, 0.01, 0.001, 0.0001]
C_list  = [10, 1, 0.1, 0.01, 0.001, 0.0001]
results = {}

best_val_f_score = -1

for i in range(len(lr_list)):
	for j in range(len(C_list)):
		lr = lr_list[i]
		C  = C_list [j]

		output = np.zeros((num_folds,3))

		for f in range(num_folds):
			train_data  = train_split_data [f]
			train_label = train_split_label[f]
			test_data   = test_split_data  [f]
			test_label  = test_split_label [f]

			classifier = SVM(max_col_prior, C)
			classifier.init_zeros()
		
			# Start training 
			for e in range(epoch_cv):
				for l in range(train_label.shape[0]):
					x = train_data[l]
					y = train_label[l]
					lr_t = learning_rate.decay_lr(lr, e)

					classifier.update(lr_t, x, y)
 
			# Finally get the predictions and accuracy
			test_predict = classifier.predict(test_data)
			output[f]    = util.get_f_scores(test_label, test_predict)
	
		# Average the predictions across folds
		temp = np.mean(output,axis=0)		
		print ("Averaged F-score = {:.4f}".format(temp[2]))		
		results[(lr, C)] = temp

		if (results[(lr, C)][2] > best_val_f_score):
			best_lr = lr
			best_C  = C
			best_val_f_score = results[(lr, C)][2]
		
# Print out results.
for lr, C in sorted(results):
	out = results[(lr, C)]
	print('lr= {:.2f} C= {:.2f} Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f} '.format(lr, C, out[0], out[1], out[2]))

print("Best lr= {:.5f} C= {:.1f}".format(best_lr, best_C)) 
print('\n\nBest f-score achieved during {}-fold cross validation: {:.4f}'.format(num_folds, best_val_f_score))

#best_C  = 10
#best_lr = 0.1

# Train on the best hyperparameter
classifier = SVM(max_col_prior, best_C)


# Start training 
for e in range(epoch_cv):
	for l in range(label_tr.shape[0]):
		x = data_tr [l]
		y = label_tr[l]
		lr_t = learning_rate.decay_lr(best_lr, e)
		classifier.update(lr_t, x, y)

# Finally get the predictions and accuracy
test_predict = classifier.predict(data_te)
out          = util.get_f_scores (label_te, test_predict)
print ("Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f}".format(out[0], out[1], out[2]))

################################################################################
# Logistic Regression with decaying learning rate
################################################################################
print("\n************************************************************************")
print("************ Logistic Regression (Decaying learning rate) **************")
print("************************************************************************")


lr_list       = [1,   0.1, 0.01, 0.001, 0.0001, 0.00001]
sigmaSq_list  = [1,   10,   100,   1000, 10000]
results       = {}
best_val_f_score = -1

for i in range(len(lr_list)):
	for j in range(len(sigmaSq_list)):
		lr = lr_list[i]
		C  = sigmaSq_list[j]
		print("lr= %f, sig = %f" %(lr,C))
		output  = np.zeros((num_folds,3))

		for f in range(num_folds):
			train_data  = train_split_data [f]
			train_label = train_split_label[f]
			test_data   = test_split_data  [f]
			test_label  = test_split_label [f]

			classifier = LogisticRegression(max_col_prior, C)
			classifier.init_zeros()
		
			# Start training 
			for e in range(epoch_cv):
				for l in range(train_label.shape[0]):
					x = train_data [l]
					y = train_label[l]
					lr_t = learning_rate.decay_lr(lr, e)

					classifier.update(lr_t, x, y)
 
			# Finally get the predictions and accuracy
			test_predict = classifier.predict(test_data)
			output[f]    = util.get_f_scores(test_label, test_predict)
	
		# Average the predictions across folds
		temp = np.mean(output,axis=0)		
		print ("Averaged F-score = {:.4f}".format(temp[2]))		
		results[(lr, C)] = temp

		if (results[(lr, C)][2] > best_val_f_score):
			best_lr = lr
			best_sigmaSq = C
			best_val_f_score = results[(lr, C)][2]
		
# Print out results.
for lr, sigmaSq in sorted(results):
	out = results[(lr, sigmaSq)]
	print('lr= {:.5f} sigmaSq= {:.1f} Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f} '.format(lr, sigmaSq, out[0], out[1], out[2]))

print("Best lr= {:.5f} sigmaSq= {:.1f}".format(best_lr, best_sigmaSq)) 
print('\n\nBest f-score achieved during {}-fold cross validation: {:.4f}'.format(num_folds, best_val_f_score))


# Train on the best hyperparameter
classifier = LogisticRegression(max_col_prior, best_sigmaSq)


# Start training 
for e in range(epoch_cv):
	for l in range(label_tr.shape[0]):
		x = data_tr [l]
		y = label_tr[l]
		lr_t = learning_rate.decay_lr(best_lr, e)
		classifier.update(lr_t, x, y)

# Finally get the predictions and accuracy
test_predict = classifier.predict(data_te)
out          = util.get_f_scores (label_te, test_predict)
print ("Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f}".format(out[0], out[1], out[2]))


################################################################################
# Naive Bayes
################################################################################
print("\n************************************************************************")
print("*************************** Naive Bayes ********************************")
print("************************************************************************")

lambda_list   = [0.5, 1, 1.5, 2.0]
results       = {}
best_val_f_score = -1

for i in range(len(lambda_list)):
		lambda_smooth = lambda_list[i]
		output = np.zeros((num_folds,3))

		for f in range(num_folds):
			# print("Fold " + str(f))
			train_data  = train_split_data [f]
			train_label = train_split_label[f]
			test_data   = test_split_data  [f]
			test_label  = test_split_label [f]

			classifier = NaiveBayes(max_col_prior, smoothing = lambda_smooth)
		
			# Start training 
			for l in range(train_label.shape[0]):
				x = train_data[l]
				y = train_label[l]
				classifier.update(x, y)
 			
			# print("Testing")
			# Finally get the predictions and accuracy
			test_predict = classifier.predict(test_data)
			output[f]    = util.get_f_scores(test_label, test_predict)
	
		# Average the predictions across folds
		temp = np.mean(output,axis=0)
		print ("Averaged F-score = {:.4f}".format(temp[2]))		
		results[(lambda_smooth)] = temp

		if (results[(lambda_smooth)][2] > best_val_f_score):
			best_smooth = lambda_smooth
			best_val_f_score = results[(lambda_smooth)][2]
		
# Print out results.
for lambda_smooth in sorted(results):
	out = results[(lambda_smooth)]
	print('lambda= {:.2f} Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f} '.format(lambda_smooth, out[0], out[1], out[2]))

print('\nBest lambda = {:.2f} Best f-score achieved during {}-fold cross validation: {:.4f}'.format(best_smooth, num_folds, best_val_f_score))

# best_smooth = 0.5

# Train on the best hyperparameter
classifier = NaiveBayes(max_col_prior, smoothing = best_smooth)

# Start training 
for l in range(train_label.shape[0]):
	x = data_tr [l]
	y = label_tr[l]
	classifier.update(x, y)

# Finally get the predictions and accuracy
test_predict = classifier.predict(data_te)
out          = util.get_f_scores (label_te, test_predict)
print ("Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f}".format(out[0], out[1], out[2]))


################################################################################
# Linear SVMs on top of decision trees
################################################################################
print("\n************************************************************************")
print("*********** Linear SVMs on top of decision trees ***********************")
print("************************************************************************")

num_trees        = 200

depth_list       = [5]#[10,20,30]
lr_list          = [    1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
C_list           = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
results          = {}
best_val_f_score = -1


for d in range(len(depth_list)):
	for k in range(len(lr_list)):
		for j in range(len(C_list)):
			depth = depth_list[d]
			lr    = lr_list[k]
			C     = C_list [j]

			output = np.zeros((num_folds,3))

			for f in range(num_folds):
				train_data  = train_split_data [f]
				train_label = train_split_label[f]
				test_data   = test_split_data  [f]
				test_label  = test_split_label [f]

								
				#train_data  = train_data [0:100]
				#train_label = train_label[0:100]
				#test_data   = test_data  [0:100]
				#test_label  = test_label [0:100] 
				

				data_tr_svm = np.zeros([train_data.shape[0], num_trees])
				data_te_svm = np.zeros([test_data .shape[0], num_trees])

				# Prepare the header
				header = np.array(['label'])
				for i in range(max_col_prior):
					header = np.concatenate([header,[str(i+1)]])
				#print(header)

				# Full data for feature transformation using trees.
				label_data_tr_full     = np.hstack((train_label.reshape((-1,1)),train_data)).astype('str')
				label_data_tr_full     = np.vstack((header,label_data_tr_full))
				label_data_tr_full_obj = Data(data = label_data_tr_full)

				# Test data for trees.
				label_data_te          = np.hstack((test_label .reshape((-1,1)),test_data)).astype('str')
				label_data_te          = np.vstack((header,label_data_te))
				label_data_te_obj      = Data(data = label_data_te)
				
				# Get the trees now
				for i in range(num_trees):
					# Get train data since train data varies for each tree
					num_examples_to_sample = np.round(0.1 * train_data.shape[0]).astype(int)
					ind = np.random.randint(0,train_data.shape[0], num_examples_to_sample)

					label_data_tr = np.hstack((train_label[ind].reshape((-1,1)),train_data[ind])).astype('str')
					label_data_tr = np.vstack((header,label_data_tr))
					train_data_obj = Data(data = label_data_tr)

					myTree = Tree()
					tree_util.ID3(train_data_obj, train_data_obj.attributes, myTree.get_root(), myTree, True, depth)
	
					# Prediction from each tree on all of train data and test data	
					out_tr    = tree_util.prediction_label(label_data_tr_full_obj, myTree)	
					out_te    = tree_util.prediction_label(label_data_te_obj     , myTree)

					data_tr_svm[:,i] = out_tr
					data_te_svm[:,i] = out_te


				# Train on the best hyperparameter
				classifier = SVM(num_trees, C)

				# SVM part start training 
				for e in range(epoch_cv):
					lr_t = learning_rate.decay_lr(lr, e)
					for l in range(train_label.shape[0]):
						x = data_tr_svm [l]
						y = train_label [l]
						classifier.update(lr_t, x, y)

				# Finally get the predictions and accuracy
				test_predict = classifier.predict(data_te_svm)
				output[f]    = util.get_f_scores (test_label, test_predict)
			
			# Average the predictions across folds
			temp = np.mean(output,axis=0)
			print ("Averaged F-score = {:.4f}".format(temp[2]))		
			results[(depth, lr, C)] = temp

			if (results[(depth, lr, C)][2] > best_val_f_score):
				best_depth = depth				
				best_lr    = lr
				best_C     = C
				best_val_f_score = results[(depth, lr, C)][2]

		
# Print out results.
for depth, lr, C in sorted(results):
	out = results[(depth, lr, C)]
	print('depth= {} lr= {:.5f} C= {:.4f} Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f}'.format(depth, lr, C, out[0], out[1], out[2]))

print('\n\nBest f-score achieved during {}-fold cross validation: {:.4f}'.format(num_folds, best_val_f_score))
print("Best depth = {} lr= {:.5f} sigmaSq= {:.1f}".format(best_depth, best_lr, best_C))


# Train on the best hyperparameter
#best_depth  = 5
#best_lr     = 0.01
#best_C      = 10


data_tr_svm = np.zeros([data_tr.shape[0], num_trees])
data_te_svm = np.zeros([data_te.shape[0], num_trees])

# Prepare the header
header = np.array(['label'])
for i in range(max_col_prior):
	header = np.concatenate([header,[str(i+1)]])
#print(header)

# Train data for trees.
label_data_tr_full     = np.hstack((label_tr.reshape((-1,1)),data_tr)).astype('str')
label_data_tr_full     = np.vstack((header,label_data_tr_full))
label_data_tr_full_obj = Data(data = label_data_tr_full)

# Test data for trees.
label_data_te          = np.hstack((label_te.reshape((-1,1)),data_te)).astype('str')
label_data_te          = np.vstack((header,label_data_te))
label_data_te_obj      = Data(data = label_data_te)

for i in range(num_trees):
	print ("Tree " + str(i+1))
	# Get train data since train data varies for each tree
	num_examples_to_sample = np.round(0.1 * data_tr.shape[0]).astype(int)
	ind = np.random.randint(0,data_tr.shape[0], num_examples_to_sample)

	label_data_tr = np.hstack((label_tr[ind].reshape((-1,1)),data_tr[ind])).astype('str')
	label_data_tr = np.vstack((header,label_data_tr))

	# Get train data and test data objects
	train_data_obj = Data(data = label_data_tr)

	myTree = Tree()
	tree_util.ID3(train_data_obj, train_data_obj.attributes, myTree.get_root(), myTree, True, best_depth)
	
	# Prediction from each tree on all of train data	
	out_tr    = tree_util.prediction_label(label_data_tr_full_obj, myTree)	
	out_te    = tree_util.prediction_label(label_data_te_obj     , myTree)

	data_tr_svm[:,i] = out_tr
	data_te_svm[:,i] = out_te


# Train on the best hyperparameter
classifier = SVM(num_trees, best_C)

# SVM part start training
for e in range(epoch_cv):
	lr_t = learning_rate.decay_lr(best_lr, e)
	for l in range(label_tr.shape[0]):
		x = data_tr_svm [l]
		y = label_tr    [l]
		classifier.update(lr_t, x, y)

# Finally get the predictions and accuracy
test_predict = classifier.predict(data_te_svm)
out          = util.get_f_scores (label_te, test_predict)
print ("Precision= {:.4f} Recall= {:.4f}  F-score= {:.4f}".format(out[0], out[1], out[2]))

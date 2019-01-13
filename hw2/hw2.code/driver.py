import util
import learning_rate
from signum import sign
from perceptron import Perceptron

import numpy as np
import math
from   random import randint
import matplotlib.pyplot as plt

epoch_cv = 10
epoch_test = 20
limit_min = 50 
limit_max = 80

np.random.seed(231)
################################################################################
# Load CV Data
################################################################################
DATA_DIR = 'dataset/CVSplits/'
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
# Load Train, Dev and Test Data
################################################################################
DATA_DIR = 'dataset/'
data_tr, label_tr, max_col_prior = util.load_file(DATA_DIR + 'diabetes.train', max_col_prior)
data_dv, label_dv, max_col_prior = util.load_file(DATA_DIR + 'diabetes.dev'  , max_col_prior)
data_te, label_te, max_col_prior = util.load_file(DATA_DIR + 'diabetes.test' , max_col_prior)


################################################################################
# Q4.4.2 Majority Baseline
################################################################################
print("\n************************************************************************")
print("*************************** Majority Label *****************************")
print("************************************************************************")

(values,counts)      = np.unique(label_tr,return_counts=True)
majority_label       = values[np.argmax(counts)]

prediction      = np.ones(label_dv.shape)*majority_label
print ("Accuracy for majority label on dev set  = %.2f" %(util.get_accuracy(label_dv, prediction)))

prediction      = np.ones(label_te.shape)*majority_label
print ("Accuracy for majority label on test set = %.2f" %(util.get_accuracy(label_te, prediction)))

################################################################################
# Simple Perceptron with fixed learning rate
################################################################################
print("\n************************************************************************")
print("************* Simple Perceptron (Fixed learning rate) ******************")
print("************************************************************************")

lr = [1, 0.1, 0.01]
acc = np.zeros((len(lr),num_folds))

for i in range(len(lr)):

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

		myPerceptron = Perceptron(max_col_prior)
		myPerceptron.init_random()
		

		# Start training the Perceptron
		for k in range(epoch_cv):
			for l in range(train_label.shape[0]):
				x = train_data[l]
				y = train_label[l]
				if(myPerceptron.predict_train(x)*y <= 0):
					myPerceptron.update(lr[i], x, y)

		# Finally get the predictions and accuracy
		test_predict = myPerceptron.predict(test_data)
		acc[i][j]    = util.get_accuracy(test_label, test_predict)	
		
# Calculate mean and standard deviations	
m = np.mean(acc,axis=1)
s = np.std (acc,axis=1)

print("lr \t Acc(mean) \t Acc(std)")
for i in range(len(lr)):
	print("%.2f \t %.2f \t\t %.2f" %(lr[i],m[i],s[i]))

# Choose the best learning rate
lr_best = lr[np.argmax(m)]

print("Best learning rate = " + str(lr_best))

# Train on the best hyperparameter
myPerceptron = Perceptron(max_col_prior)
myPerceptron.init_random()
num_update = 0
acc_dv_best = 0
acc_dv_list = []
acc_te_list = []

for k in range(epoch_test):
	for l in range(label_tr.shape[0]):
		x = data_tr[l]
		y = label_tr[l]
		if(myPerceptron.predict_train(x)*y <= 0):
			myPerceptron.update(lr_best, x, y)
			num_update += 1

	# Finally get the predictions and accuracy
	predict_dv = myPerceptron.predict(data_dv)
	acc_dv = util.get_accuracy(label_dv, predict_dv)
	acc_dv_list.append(acc_dv)
	
	predict_te = myPerceptron.predict(data_te)
	acc_te = util.get_accuracy(label_te, predict_te)
	acc_te_list.append(acc_te)

print("Total number of updates = " + str(num_update))

acc_dv_list = np.array(acc_dv_list)

# Get the best development and test accuracy
ind = np.argmax(acc_dv_list)
print("\nAccuracy on best hyperparameter\ndev  = %.2f\ntest = %.2f" %(acc_dv_list[ind], acc_te_list[ind]))

# Plot the learning curves
plt.figure()
plt.ylim([limit_min, limit_max])
plt.plot(range(1,epoch_test+1),acc_dv_list,'-o')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Simple Perceptron (Fixed Learning Rate)')
plt.grid()
print('Saving the learning curve to simple.png')
plt.savefig('simple.png')


################################################################################
# Simple Perceptron with decaying learning rate
################################################################################
print("\n************************************************************************")
print("************* Simple Perceptron (Decaying learning rate) ***************")
print("************************************************************************")

lr = [1, 0.1, 0.01]
acc = np.zeros((len(lr),num_folds))

for i in range(len(lr)):

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

		myPerceptron = Perceptron(max_col_prior)
		myPerceptron.init_random()
		
		t = 0
		# Start training the Perceptron
		for k in range(epoch_cv):
			for l in range(train_label.shape[0]):
				x = train_data[l]
				y = train_label[l]
				lr_t = learning_rate.decay_lr(lr[i], t)

				if(myPerceptron.predict_train(x)*y <= 0):
					myPerceptron.update(lr_t, x, y)	
				
				t += 1 
		# Finally get the predictions and accuracy
		test_predict = myPerceptron.predict(test_data)
		acc[i][j]    = util.get_accuracy(test_label, test_predict)	
		
# Calculate mean and standard deviations	
m = np.mean(acc,axis=1)
s = np.std (acc,axis=1)

print("lr \t Acc(mean) \t Acc(std)")
for i in range(len(lr)):
	print("%.2f \t %.2f \t\t %.2f" %(lr[i],m[i],s[i]))

# Choose the best learning rate
lr_best = lr[np.argmax(m)]

print("Best learning rate = " + str(lr_best))

# Train on the best hyperparameter
myPerceptron = Perceptron(max_col_prior)
myPerceptron.init_random()
num_update = 0
acc_dv_best = 0
acc_dv_list = []
acc_te_list = []

t = 0
for k in range(epoch_test):
	for l in range(label_tr.shape[0]):
		x = data_tr[l]
		y = label_tr[l]
		lr_t = learning_rate.decay_lr(lr_best, t)

		if(myPerceptron.predict_train(x)*y <= 0):
			myPerceptron.update(lr_best, x, y)
			num_update += 1
		
		t += 1
	# Finally get the predictions and accuracy
	predict_dv = myPerceptron.predict(data_dv)
	acc_dv = util.get_accuracy(label_dv, predict_dv)
	acc_dv_list.append(acc_dv)
	
	predict_te = myPerceptron.predict(data_te)
	acc_te = util.get_accuracy(label_te, predict_te)
	acc_te_list.append(acc_te)

print("Total number of updates = " + str(num_update))

acc_dv_list = np.array(acc_dv_list)

# Get the best development and test accuracy
ind = np.argmax(acc_dv_list)
print("\nAccuracy on best hyperparameter\ndev  = %.2f\ntest = %.2f" %(acc_dv_list[ind], acc_te_list[ind]))

# Plot the learning curves
plt.figure()
plt.ylim([limit_min, limit_max])
plt.plot(range(1,epoch_test+1),acc_dv_list,'-o')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Simple Perceptron (Decaying Learning Rate)')
plt.grid()
print('Saving the learning curve to decaying.png')
plt.savefig('decaying.png')


################################################################################
# Margin Perceptron
################################################################################
print("\n************************************************************************")
print("************ Margin Perceptron (Decaying Learning Rate) ****************")
print("************************************************************************")

mu = [1, 0.1, 0.01]
lr = [1, 0.1, 0.01]
acc = np.zeros((len(mu),len(lr),num_folds))

for m in range(len(mu)):
	for i in range(len(lr)):

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

			myPerceptron = Perceptron(max_col_prior)
			myPerceptron.init_random()
		
			t = 0
			# Start training the Perceptron
			for k in range(epoch_cv):
				for l in range(train_label.shape[0]):
					x = train_data[l]
					y = train_label[l]
					lr_t = learning_rate.decay_lr(lr[i], t)

					if(myPerceptron.predict_train(x)*y <= mu[m]):
						myPerceptron.update(lr_t, x, y)	
				
					t += 1 
			# Finally get the predictions and accuracy
			test_predict = myPerceptron.predict(test_data)
			acc[m][i][j]    = util.get_accuracy(test_label, test_predict)	
		
# Calculate mean and standard deviations
m = np.mean(acc,axis=2)
s = np.std (acc,axis=2)

print("mu \t lr \t Acc(mean) \t Acc(std)")
for i in range(len(mu)):
	for j in range(len(lr)):
		print("%.2f \t %.2f \t %.2f \t\t %.2f" %(mu[i], lr[j], m[i][j] ,s[i][j]))

# Choose the best mu and learning rate
ind = np.argwhere(m==np.max(m))
mu_best = mu[ind[0,0]]
lr_best = lr[ind[0,1]]


print("Best mu = " + str(mu_best))
print("Best learning rate = " + str(lr_best))

# Train on the best hyperparameter
myPerceptron = Perceptron(max_col_prior)
myPerceptron.init_random()
num_update = 0
acc_dv_best = 0
acc_dv_list = []
acc_te_list = []

t = 0
for k in range(epoch_test):
	for l in range(label_tr.shape[0]):
		x = data_tr[l]
		y = label_tr[l]
		lr_t = learning_rate.decay_lr(lr_best, t)

		if(myPerceptron.predict_train(x)*y <= mu_best):
			myPerceptron.update(lr_t, x, y)
			num_update += 1
		
		t += 1
	# Finally get the predictions and accuracy
	predict_dv = myPerceptron.predict(data_dv)
	acc_dv = util.get_accuracy(label_dv, predict_dv)
	acc_dv_list.append(acc_dv)
	
	predict_te = myPerceptron.predict(data_te)
	acc_te = util.get_accuracy(label_te, predict_te)
	acc_te_list.append(acc_te)

print("Total number of updates = " + str(num_update))

acc_dv_list = np.array(acc_dv_list)

# Get the best development and test accuracy
ind = np.argmax(acc_dv_list)
print("\nAccuracy on best hyperparameter\ndev  = %.2f\ntest = %.2f" %(acc_dv_list[ind], acc_te_list[ind]))

# Plot the learning curves
plt.figure()
plt.plot(range(1,epoch_test+1),acc_dv_list,'-o')
plt.ylim([limit_min, limit_max])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Margin Perceptron (Decaying Learning Rate)')
plt.grid()
print('Saving the learning curve to margin.png')
plt.savefig('margin.png')



################################################################################
# Averaged Perceptron
################################################################################
print("\n************************************************************************")
print("************************ Averaged Perceptron ***************************")
print("************************************************************************")

lr = [1, 0.1, 0.01]
acc = np.zeros((len(lr),num_folds))

for i in range(len(lr)):
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

		myPerceptron = Perceptron(max_col_prior,avg_flag = True)
		myPerceptron.init_random()
	
		t = 0
		# Start training the Perceptron
		for k in range(epoch_cv):
			for l in range(train_label.shape[0]):
				x = train_data[l]
				y = train_label[l]				
				lr_t = learning_rate.decay_lr(lr[i], t)

				if(myPerceptron.predict_train(x)*y <= 0):
					myPerceptron.update(lr_t, x, y)	
			
				# Average wt and bias everytime
				myPerceptron.update_avg()


		# Finally get the predictions and accuracy
		test_predict = myPerceptron.predict(test_data)
		acc[i][j]    = util.get_accuracy(test_label, test_predict)	
	
# Calculate mean and standard deviations
m = np.mean(acc,axis=1)
s = np.std(acc,axis=1)

print("mu \t Acc(mean) \t Acc(std)")
for i in range(len(lr)):
		print("%.2f \t %.2f \t\t %.2f" %(lr[i], m[i] ,s[i]))

# Choose the best
ind = np.argwhere(m==np.max(m))
lr_best = lr[ind[0,0]]

print("Best lr = " + str(lr_best))

# Train on the best hyperparameter
myPerceptron = Perceptron(max_col_prior,avg_flag = True)
myPerceptron.init_random()
num_update  = 0
acc_dv_best = 0
acc_dv_list = []
acc_te_list = []

t = 0
for k in range(epoch_test):
	for l in range(label_tr.shape[0]):
		x = data_tr[l]
		y = label_tr[l]				
		lr_t = lr_best

		if(myPerceptron.predict_train(x)*y <= 0):
			myPerceptron.update(lr_t, x, y)
			num_update += 1

		# Averaged it everytime
		myPerceptron.update_avg()

	# Finally get the predictions and accuracy
	predict_dv = myPerceptron.predict(data_dv)
	acc_dv     = util.get_accuracy(label_dv, predict_dv)
	acc_dv_list.append(acc_dv)
	
	predict_te = myPerceptron.predict(data_te)
	acc_te     = util.get_accuracy(label_te, predict_te)
	acc_te_list.append(acc_te)

print("Total number of updates = " + str(num_update))
acc_dv_list = np.array(acc_dv_list)

# Get the best epoch for development and test accuracy
ind = np.argmax(acc_dv_list)
print("\nAccuracy on best hyperparameter\ndev  = %.2f\ntest = %.2f" %(acc_dv_list[ind], acc_te_list[ind]))

# Plot the learning curves
plt.figure()
plt.plot(range(1,epoch_test+1),acc_dv_list,'-o')
plt.ylim([limit_min, limit_max])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Averaged Perceptron')
plt.grid()
print('Saving the learning curve to averaged.png')
plt.savefig('averaged.png')



################################################################################
# Aggressive Perceptron
################################################################################
print("\n************************************************************************")
print("********************** Aggressive Perceptron ***************************")
print("************************************************************************")

mu = [1, 0.1, 0.01]
acc = np.zeros((len(mu),num_folds))

for m in range(len(mu)):

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

		myPerceptron = Perceptron(max_col_prior)
		myPerceptron.init_random()
	
		t = 0
		# Start training the Perceptron
		for k in range(epoch_cv):
			for l in range(train_label.shape[0]):
				x = train_data[l]
				y = train_label[l]

				if(myPerceptron.predict_train(x)*y <= mu[m]):						
					lr_t = learning_rate.aggressive_lr(mu[m], x, y, myPerceptron.get_weight())
					myPerceptron.update(lr_t, x, y)	
			
				t += 1 
		# Finally get the predictions and accuracy
		test_predict = myPerceptron.predict(test_data)
		acc[m][j]    = util.get_accuracy(test_label, test_predict)	
	
# Calculate mean and standard deviations
m = np.mean(acc,axis=1)
s = np.std(acc,axis=1)

print("mu \t Acc(mean) \t Acc(std)")
for i in range(len(mu)):
		print("%.2f \t %.2f \t\t %.2f" %(mu[i], m[i] ,s[i]))

# Choose the best
ind = np.argwhere(m==np.max(m))
mu_best = mu[ind[0,0]]

print("Best mu = " + str(mu_best))

# Train on the best hyperparameter
myPerceptron = Perceptron(max_col_prior)
myPerceptron.init_random()
num_update  = 0
acc_dv_best = 0
acc_dv_list = []
acc_te_list = []

t = 0
for k in range(epoch_test):
	for l in range(label_tr.shape[0]):
		x = data_tr[l]
		y = label_tr[l]

		if(myPerceptron.predict_train(x)*y <= mu_best):	
			lr_t = learning_rate.aggressive_lr(mu_best, x, y, myPerceptron.get_weight())
			myPerceptron.update(lr_t, x, y)
			num_update += 1

	# Finally get the predictions and accuracy
	predict_dv = myPerceptron.predict(data_dv)
	acc_dv     = util.get_accuracy(label_dv, predict_dv)
	acc_dv_list.append(acc_dv)
	
	predict_te = myPerceptron.predict(data_te)
	acc_te     = util.get_accuracy(label_te, predict_te)
	acc_te_list.append(acc_te)

print("Total number of updates = " + str(num_update))
acc_dv_list = np.array(acc_dv_list)

# Get the best epoch for development and test accuracy
ind = np.argmax(acc_dv_list)
print("\nAccuracy on best hyperparameter\ndev  = %.2f\ntest = %.2f" %(acc_dv_list[ind], acc_te_list[ind]))

# Plot the learning curves
plt.figure()
plt.plot(range(1,epoch_test+1),acc_dv_list,'-o')
plt.ylim([limit_min, limit_max])
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Aggressive Perceptron')
plt.grid()
print('Saving the learning curve to aggresive.png')
plt.savefig('aggressive.png')

import numpy as np
import os
import math

def get_pred(X):
	out = np.sign(X[:,0])
	return out

def get_pred_2(X):
	out = np.sign(X[:,0]-2)
	return out

def get_pred_3(X):
	out = -np.sign(X[:,0])
	return out

def get_pred_4(X):
	out = -np.sign(X[:,1])
	return out


def error(p,y,D):
	temp = np.multiply(p,y)
	temp[temp == 1] = 0
	temp[temp == -1] = 1
	temp = np.multiply(D,temp)	
	return np.sum(temp)

def get_outputs(X,Y,T,f):
	print("\n\n-------------------------------------------------------------------")	
	n = X.shape[0]
	D = 1/n*np.ones((n,))

	for i in range(T):
		print("\nTime = %d" %(i))
		print("D = ")
		print(D)
		p    = f[i](X)
		#print(Y)		
		print("predictions = ")		
		print(p)
		err  = error(p,Y,D)
		
		if(err >= 0.5):
			#Wrong Hypothesis
			print("Error = " + str(err))
			print("Wrong hypothesis chosen !!! Aborting")
			return
		
		alpha = 0.5*math.log( (1-err)/err)
		D_temp = np.multiply(D, np.exp(-alpha* np.multiply(Y, p)))
		Z = np.sum(D_temp)
		D = D_temp/Z
	
		print("Error = " + str(err))
		print("Alpha = " + str(alpha))			
		print("Z = " + str(Z))	


X = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
Y = np.array([-1,1,-1,-1])

T=4


#get_outputs(X,Y,T,[get_pred,get_pred_2,get_pred,get_pred_2])
#get_outputs(X,Y,T,[get_pred,get_pred_2,get_pred,get_pred_4])
#get_outputs(X,Y,T,[get_pred,get_pred_2,get_pred,get_pred_4])
#get_outputs(X,Y,T,[get_pred,get_pred_2,get_pred_4,get_pred_2])

get_outputs(X,Y,T,[get_pred,get_pred_2,get_pred_4,get_pred])


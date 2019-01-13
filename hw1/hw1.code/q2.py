import numpy as np
from numpy import array

def foo():
	a = array([[1,1,1,1,1],[0,1,1,1,1],[1,1,0,0,-1],[1,0,0,0,-1],[0,0,0,0,-1],[0,0,0,1,-1],[0,0,1,0,-1],[0,0,1,1,-1],[1,0,1,1,1],[1,1,0,1,1],[0,1,0,0,-1],[0,1,0,1,-1],[0,1,1,0,-1],[1,0,0,1,-1],[1,0,1,0,-1],[1,1,1,0,1]])

	print(a.shape)

	ground = a[:,4].astype(int)
	print(ground)

	# Adding 1 to the data
	data = a[:,0:4]
	bias_1 = np.ones((a.shape[0],1))
	data = np.append(data,bias_1,axis=1)
	print(data)

	w_range = [0,0.5,1]
	b_range = [-3.5,-3.25,-3,-2.75,-2.5,-2.25,-2,-1.75,-1.5,-1.25,-1,0.75,-0.5,0.25,0,0.5,1]
	cnt = 0;

	for i in range(len(w_range)):
		for j in range(len(w_range)):
			for k in range(len(w_range)):
				for l in range(len(w_range)):
					for m in range(len(b_range)):
						w = np.transpose(array([[w_range[i],w_range[j],w_range[k],w_range[l],b_range[m]]]))
						#print(np.transpose(w))						
						pred = (np.matmul(data,w))
						pred[pred>=0] = 1
						pred[pred<0] = -1 						
						pred = np.transpose(pred)[0].astype(int);						
						cnt = cnt + 1
						if(cnt % 100 == 0):						
							print(cnt)
							#return						
						if ((pred.astype(int)==ground.astype(int)).all()):
							print("Found a weight combination")
							print(pred)
							print(ground)
							print(np.transpose(w))	
							return

foo()

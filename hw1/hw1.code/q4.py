import numpy as np
import math

def ent(a,b):
	p = float(a)/(a+b)
	h = -p*math.log(p,2) - (1-p)*math.log(1-p,2)
	return h

def new_crit(h_orig,h,cost,scale=1.0):
	g = h_orig-scale*h
	
	g1 = g*g/cost
	g2 = (2**(g)-1)/math.sqrt(cost+1)
	print (g,g1,g2)

h_orig = ent(7,5)

h_shape = ent(2,1)
new_crit(h_orig,h_shape,10)

h_color = 2*(2/12)*ent(1,1) + (5/12)*ent(4,1) + (3/12)*ent(1,2)
new_crit(h_orig,h_color,30)

h_size = ent(3,1)
new_crit(h_orig,h_size,50)

h_mat = 2*3/12*ent(2,1) + 1/3*ent(2,2)
new_crit(h_orig,h_mat,100)

"""
0.06157292259666325 0.0003791224797094684 0.01314678303321537
0.1161580919344628 0.0004497567440618371 0.015058957894028608
0.16859063219201997 0.0005684560252580993 0.017357865267819364
0.18738750629057477 0.00035114077513800196 0.013801150058785192
"""

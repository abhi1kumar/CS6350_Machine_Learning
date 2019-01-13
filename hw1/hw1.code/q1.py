import numpy as np
import math

def ent(a,b):
	p = float(a)/(a+b)
	h = -p*math.log(p,2) - (1-p)*math.log(1-p,2)
	return h

def imp(a,b):
	h = max(a,b);
	h = 1 - float(h)/(a+b);
	return h

# 2(c)
h_t = ent(4,4)

h_var = 4/8*ent(2,2) + 3/8*ent(1,2) + 1/8*0
print(h_t-h_var)

h_col = 3/8*ent(2,1) + 3/8*ent(1,2) + 2/8*ent(1,1)
print(h_t-h_col)

h_sml = 4/8*ent(1,3) + 4/8*ent(3,1)
print(h_t-h_sml)

h_time = 5/8*ent(2,3) + 3/8*ent(2,1)
print(h_t-h_time)

"""
0.15563906222956647
0.06127812445913283
0.18872187554086717
0.04879494069539847
"""

h_t = imp(4,4)

h_var = 4/8*imp(2,2) + 3/8*imp(1,2) + 1/8*0
print(h_t-h_var)

h_col = 3/8*imp(2,1) + 3/8*imp(1,2) + 2/8*imp(1,1)
print(h_t-h_col)

h_sml = 4/8*imp(1,3) + 4/8*imp(3,1)
print(h_t-h_sml)

h_time = 5/8*imp(2,3) + 3/8*imp(2,1)
print(h_t-h_time)


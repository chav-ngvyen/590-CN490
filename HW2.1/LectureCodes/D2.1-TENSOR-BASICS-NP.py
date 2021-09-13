

#CODE MODIFIED FROM:
# chollet-deep-learning-in-python

import numpy as np
from pandas import DataFrame
import time

#------------------------
#EXPLORE IMAGE
#------------------------
ipause=False 
#QUICK INFO ON NP ARRAY
def get_info(X,MESSAGE='SUMMARY'):
	print("\n------------------------")
	print(MESSAGE)
	print("------------------------")
	print("TYPE:",type(X))

	if(str(type(x))=="<class 'numpy.ndarray'>"):

		print("SHAPE:",X.shape)
		print("MIN:",X.min())
		print("MAX:",X.max())
		#NOTE: ADD SLICEING
		print("DTYPE:",X.dtype)
		print("NDIM:",X.ndim)
		print("IS CONTIGUOUS:",X.data.contiguous)
		#PRETTY PRINT 
		if(X.ndim==1 or X.ndim==2 ): 
			print("MATRIX:")
			print(DataFrame(X).to_string(index=False, header=False))
			# print("EDGES ARE INDICES: i=row,j=col") 
			# print(DataFrame(X)) 	
		if(ipause):  time.sleep(3)
	else:
		print("ERROR: INPUT IS NOT A NUMPY ARRAY")

#SCALAR (0D TENSOR)
x = np.array(10); get_info(x)

#VECTOR AS 1D ARRARY
x = np.array([12, 3, 6, 14]); get_info(x)
# exit()
#VECTOR AS 2D ARRAY 
x = np.array([12, 3, 6, 14]);  x=x.reshape(len(x),1); get_info(x) #COLUMN VECTOR
x = np.array([12, 3., 6, 14]); x=x.reshape(1,len(x)); get_info(x) #ROW VECTOR
# exit()
#MATRIX (2D TENSOR)
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]); get_info(x)
# exit()
# #3D TENSOR
x = np.array([[[5., 78, 2, 34, 0],
			   [6, 79, 3, 35, 1],
			   [7, 80, 4, 36, 2]],
			  [[5, 78, 2, 34, 0],
			   [6, 79, 3, 35, 1],
			   [7, 80, 4, 36, 2]],
			  [[5, 78, 2, 34, 0],
			   [6, 79, 3, 35, 1],
			   [7, 80, 4, 36, 2]]]) ; get_info(x,"3D TENSOR")
# Note there was no pretty print because ndim is not 1 or 2
# exit()

# #TRANSPOSE
x = np.array([[11, 12, 13],
            [21, 22, 23]]); 
get_info(x, "BEFORE TRANSPOSE")
get_info(np.transpose(x), "AFTER  TRANSPOSE")


# exit()
#SLICING
x = np.array([[11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
              [41, 42, 43, 44], 
              [51, 52, 53, 54]]); 

#NOTICE HOW ITS INCLUSIVE ON THE LEFT
#AND EXCLUSIVE ON THE RIGHT
get_info(x, "BEFORE SLICING")
# exit()

##ll rows, column index 1 (so second column)
get_info(x[:,1], 	"SLICE-1: x[:,1]")
# exit()

## All columns, row index 2 (so third row)
get_info(x[2,:], 	"SLICE-2: x[2,:]")
# exit()

##et_info(x[1:3], 	"SLICE-3: x[1:3]")
# exit()

get_info(x, "BEFORE SLICING")

## All rows, column index starting -3 to right before -1
get_info(x[:,-3:-1],"SLICE-4: x[:,-3:-1]")
# exit()

## All rows, column index from 0 to column before 2
get_info(x[:,0:2], 	"SLICE-5: x[:,0:2]")
# exit()

#BROADCAST
get_info(x, "BEFORE BROADCAST")
get_info(x+1000, "ADD 1000 TO ALL")
get_info(x+x[0,:], "ADD FIRST ROW TO EACH ROW")
# exit()

#RESHAPING 
get_info(x.reshape(x.shape[0]*x.shape[1],1), "x.reshape(x.shape[0]*x.shape[1],1)")
## x.shape is (5,4), so x.shape[0] is 5 and x.shape[1] is 4
## -> essentially this means x.reshape(20,1) 
# exit()

get_info(x.reshape(1,x.shape[0]*x.shape[1]), "x.reshape(1,x.shape[0]*x.shape[1])")
## similarly, this means x.reshape(1,20)
# exit()

print(x.shape[0]*x.shape[1]/2)
print(int(x.shape[0]*x.shape[1]/2))
 
get_info(x.reshape(int(x.shape[0]*x.shape[1]/2),2), "x.reshape(int(x.shape[0]*x.shape[1]/2),2)")

exit()
# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING

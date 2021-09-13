import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

# print(X)
# exit()

# Before transposing, X is a list of len 3
print(type(X), len(X))

print(X[0])
# print(X[1])
# print(X[2])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

# After transposing, x is a numpy array
print(type(X))
# With 2 dimensions
print(X.ndim)
# 250 rows, 3 columns
print(X.shape)



#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]
# xcol and ycol index the x and y columns used in the model
print(xcol, ycol)

print(type(y))
print(y.shape)


#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 

# x[:] is a copy of the x, <18 returns a boolean for x<18

print(y.shape)
print(x.shape)

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)
# Create the original mean and standard deviation of x & y to use later for un-normalize

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION

f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])

# x is a 1D vector
print(x.shape)
print(x.ndim)

# x.shape[0] is the number of rows 
print(x.shape[0])
# so np.random.permutation(x.shape[0]) is just np.random.permutation(39)

# Round the result of (0.8 * 31)
CUT1=int(f_train*x.shape[0]); 

print(type(rand_indices))
print(rand_indices.ndim)
print(rand_indices.shape)

# Slice from beginning to index CUT1, then from index CUT1 to the end
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]

print(type(train_idx))
print(train_idx)
print(val_idx)
# then subset x and y based on the index from train_idx and val_idx
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p):
	global iterations,loss_train,loss_val,iteration

	#TRAINING LOSS
	#y_predict if run model on x train
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	#y_predict if run model on x validation
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	if(iteration==0):    print("iteration	training_loss	validation_loss") 
	if(iteration%25==0): print(iteration,"	",training_loss,"	",validation_loss) 
	
	#RECORD FOR PLOTING
	loss_train.append(training_loss); loss_val.append(validation_loss)
	iterations.append(iteration); iteration+=1

	return training_loss

#INITIAL GUESS
po=np.random.uniform(0.5,1.,size=NFIT)
# size is the number of element in the array

#TRAIN MODEL USING SCIPY MINIMIZ 
res = minimize(loss, po, method=OPT_ALGO, tol=1e-15);  popt=res.x
print(res)
print(type(res))
print("OPTIMAL PARAM:",popt)

#PREDICTIONS
print(xt)
print(sorted(xt))
print(type(sorted(xt)))

# sort x train and convert that into a numpy array (sorted list)
xm=np.array(sorted(xt))

print(np.array(sorted(xt)))
print(type(xm))

# put xm and popt into the model function
yp=np.array(model(xm,popt))


#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()

# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING
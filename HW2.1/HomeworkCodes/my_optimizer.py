import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

######################################################
# A lot of this was copied from HW1.1-SciPy-BASIC.py #
######################################################

#USER PARAMETERS
IPLOT=True
INPUT_FILE='../LectureCodes/weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']

OPT_ALGO='BFGS'

##############################
#Parameters for my_minimizer #
##############################
# PARADIGM = 'batch'
PARADIGM = 'mini-batch'
# PARADIGM = 'stochastic'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
#model_type="logistic"; NFIT=4; xcol=1; ycol=2;
model_type="linear";   NFIT=2; xcol=1; ycol=2; 
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
# print(type(X))
# # With 2 dimensions
# print(X.ndim)
# # 250 rows, 3 columns
# print(X.shape)



#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]
# xcol and ycol index the x and y columns used in the model
# print(xcol, ycol)

# print(type(y))
# print(y.shape)


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
x_train=x[train_idx]; y_train=y[train_idx]; x_val=x[val_idx];   y_val=y[val_idx]

#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#####################
# My loss function #
#####################

iteration=0; iterations=[]; loss_train=[];  loss_val=[]

def my_loss():
    global iterations,loss_train,loss_val,iteration
    if (PARADIGM == 'batch'):
        xt = x_train
    if (PARADIGM == 'mini-batch'):
        HALF = int(0.5*x_train.shape[0])
        xt_rand_idx = np.random.permutation(x_train.shape[0])
        x_half1 = xt_rand_idx[:HALF]
        xt = x_half1
    


my_loss()

exit()

    # # Training loss
    # y_pred = model(xt, p)
    # training_loss = (np.mean((y_pred - y_train)**2.0))

    # # Validation loss:
    # y_pred  = model(x_val, p )
    # validation_loss = (np.mean(y_pred - y_val)**2.0)
    	
    # # Append to the 3 empty lists    
    # loss_train.append(training_loss)
    # loss_val.append(validation_loss)
    # iterations.append(iteration)
    
    # # Add 1 to iteration
    # iteration+=1
    
    # return training_loss


#############################33
po=np.random.uniform(0.5,1.,size=NFIT)
#TRAIN MODEL USING SCIPY MINIMIZE
res = minimize(my_loss, po, method=OPT_ALGO, tol=1e-15);  popt=res.x

exit()
##############
# Batch method

def batch():
    if (PARADIGM == 'batch'):
        xt = x_train
        my_loss(xt, p)
    
    if (PARADIGM == 'mini-batch'):
        xt_half1 = 0.5*x_train.shape[0]
        loss(xt_half1, p) 

exit()
##############

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
print(len(po))

#TRAIN MODEL USING SCIPY MINIMIZE
res = minimize(loss, po, method=OPT_ALGO, tol=1e-15);  popt=res.x

################
# My optimizer #
################

# Copied from Prof Hickman's email


def my_minimizer(f, x0, algo = 'MOM', LR = 0.05):
    '''
    f: function to minimize
    x0: initial guess
    algo: 'MOM' or 'GD'
    LR: learning rate 
    '''

    # PARAM
    dx = 0.0001         # Step size for finite difference
    t = 0               # Initial iteration counter
    tmax = 3000         # Max number of iterations
    tol = 10**-30       # Exit after change if f is less than this
    ICLIP = False

    NDIM = len(x0)      # number of dimensions (columns)

    xi = x0             # Initial guess
    dx_m1 = 0           # Initialize for momentum algorithm
    alpha = 0.1         # exponential decay factor for momentum algo

    if(PARADIGM == 'stocastic'):
        LR = 0.002
        tmax = 25000
        ICLIP = True

    # CLIP GRADIENTS
    if(ICLIP):
        max_grad = 10
        if(grad_i > max_grad): grad_i = max_grad
        if(grad_i < -max_grad): grad_i = -max_grad

#######################################
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


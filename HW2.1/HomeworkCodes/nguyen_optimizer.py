import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt
import json 

#################################
# INPUT 

INPUT_FILE='../LectureCodes/weight.json'
DATA_KEYS=['x','is_adult','y']

model_type="logistic"; NFIT=4; xcol=1; ycol=2;

IPLOT = True
NORMALIZE = False

# SETTINGS
algo = "GD"
batch_size = 0.5

# PARADIGM = 'minibatch'
# PARADIGM = 'batch'
PARADIGM = 'stochastic'

############################################

# READ JSON
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)
# Create the original mean and standard deviation of x & y to use later for un-normalize

if(NORMALIZE):
	x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 


#PARTITION DATA
f_train=0.8; f_val=0.15; f_test = 0.05
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
CUT2=int((f_train+f_val)*x.shape[0]);
train_idx,  val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx];
xtest = x[test_idx]; ytest = y[test_idx]

#CREATE EMPTY CONTAINERS
loss_train = []
loss_val = []

#######################################
# DEFINE MODELS

def model(x,p):
	return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

def new_loss(p,index=None):
	global loss_train, loss_val

	xbatch = xt[index]
	ybatch = yt[index]

	yp = model(xbatch,p)
	mse=(np.mean((yp-ybatch)**2.0))  

	# the training set changes depending on PARADIGm
	loss_train.append(mse)

	# but the validation set doesn't
	ypval = model(xv,p)
	val_mse = (np.mean((ypval-yv)**2.0))
	loss_val.append(val_mse)

	return mse	

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 



# MY MINIMIZER
def my_minimizer(fun, x0, algo = "GD"):
	global loss_train, loss_val

	#General params
	dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
	LR=0.05								#LEARNING RATE
	n_iter=0 	 						#INITIAL iter COUNTER
	iter_max=5000						#MAX NUMBER OF iter
	tol=1e-15							#EXIT AFTER CHANGE IN F IS LESS THAN THIS

	# Momentum params
	dx_m1 = 0
	alpha=0

	xi = x0								#INITIAL GUESS
	NDIM = len(x0)						#dimension of optimization problem is defined by length of vector

	print("INITAL GUESS: ",xi)

	# PARADIGM SETTINGS
	if (PARADIGM == 'batch'):
		index = [range(len(xt))]

	if (PARADIGM == 'stochastic'):
		index = np.array([0])
		if (n_iter == 0):
			index += 1
			if (n_iter == len(xt)):
				index = np.array([0])
	
	if (PARADIGM == 'minibatch'):
		mini_index = np.random.permutation(len(xt))
		CUT_BATCH = int(batch_size*len(xt))
		if(n_iter%2==0):
			index = mini_index[:CUT_BATCH]
		else:
			index = mini_index[CUT_BATCH:]

	
	while(n_iter<=iter_max):
		
		#NUMERICALLY COMPUTE GRADIENT 
		df_dx=np.zeros(NDIM)
		for i in range(0,NDIM):
			dX=np.zeros(NDIM);
			dX[i]=dx; 
			xm1=xi-dX; print(xi,xm1,dX,dX.shape,xi.shape)
			df_dx[i]=(fun(xi)-fun(xm1))/dx

		if (algo == "GD"):
			xip1=xi-LR*df_dx #STEP 

		if (algo == "MOM"):
			dx_m1 = alpha*dx_m1 - LR*df_dx
			xip1 = xi -LR*df_dx + alpha*dx_m1

		if(n_iter%1==0):
			df=np.mean(np.absolute(fun(xip1)-fun(xi)))
			
			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break
					
		xi = xip1
		n_iter=n_iter+1




	return xi




po=np.random.uniform(2,1.,size=NFIT)
popt = my_minimizer(new_loss, po)


xm=np.array(sorted(xt))
ypred=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 


if(NORMALIZE == False & IPLOT):
	fig, ax = plt.subplots()
	ax.plot(xt, yt, 'o', label='Training set')
	ax.plot(xv, yv, 'x', label='Validation set')
	ax.plot(xtest, ytest, 'x', label = 'Test set')
	ax.plot(xm, ypred, '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.title(PARADIGM +" "+ algo + " (not normalized)")
	plt.legend()
	plt.show()

if(NORMALIZE & IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xtest), unnorm_y(ytest), 'x', label = 'Test set')
	ax.plot(unnorm_x(xm), unnorm_y(ypred), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	# plt.title('Input was normalized')
	plt.legend()
	plt.show()


if(IPLOT & NORMALIZE == False):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'x', label = 'Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.title(PARADIGM +" "+ algo + " (not normalized)")
	plt.legend()
	plt.show()

iter_list = range(len(loss_train))

if(IPLOT & NORMALIZE == False):
	fig, ax = plt.subplots()
	ax.plot(iter_list, loss_train, 'o', label='Training loss')
	ax.plot(iter_list, loss_val, 'x', label = 'Validation loss')
	# ax.margins(0,0.1)
	plt.xlim(0,1000)
	plt.ylim(0,2000)
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.title(PARADIGM +" "+ algo + " (not normalized)")
	plt.legend()
	plt.show()


exit()





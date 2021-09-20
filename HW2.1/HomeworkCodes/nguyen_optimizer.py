import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt
import json 
from   scipy.optimize import minimize



INPUT_FILE='../LectureCodes/weight.json'
DATA_KEYS=['x','is_adult','y']

batch_size = 0.5
PARADIGM = 'minibatch'
# PARADIGM = 'batch'
# PARADIGM = 'stochastic'


model_type="logistic"; NFIT=4; xcol=1; ycol=2;



IPLOT = True
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


#PARTITION
f_train=0.8; f_val=0.15; f_test = 0.05
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
CUT2=int((f_train+f_val)*x.shape[0]);
train_idx,  val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx];
xtest = x[test_idx]; ytest = y[test_idx]




def model(x,p):
	return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

def new_loss(p,index=None):

	xbatch = xt[index]
	ybatch = yt[index]

	yp = model(xbatch,p)
	mse=(np.mean((yp-ybatch)**2.0))  

	return mse	

epoch = 0
epochs = []

def my_minimizer(fun, x0):

	global epoch, epochs

	dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
	LR=0.05								#LEARNING RATE
	t=0 	 							#INITIAL epoch COUNTER
	tmax=1000							#MAX NUMBER OF epoch
	tol=10**-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	xi = x0
	NDIM = len(x0)
	print("INITAL GUESS: ",xi)

	if (PARADIGM == 'batch'):
		index = [range(len(xt))]

	if (PARADIGM == 'stochastic'):
		index = np.array([0])
		if (t == 0):
			index += 1
			if (t == len(xt)):
				index = np.array([0])
	
	if (PARADIGM == 'minibatch'):
		mini_index = np.random.permutation(len(xt))
		CUT_BATCH = int(batch_size*len(xt))
		if(t%2==0):
			index = mini_index[:CUT_BATCH]
		else:
			index = mini_index[CUT_BATCH:]

	# if (PARADIGM == 'stochastic'):
	# 	tmax = 2500
	# 	LR = 0.02
	# 	ICLIP = True

	
	while(t<=tmax):
		
		#NUMERICALLY COMPUTE GRADIENT 
		df_dx=np.zeros(NDIM)
		for i in range(0,NDIM):
			dX=np.zeros(NDIM);
			dX[i]=dx; 
			xm1=xi-dX; print(xi,xm1,dX,dX.shape,xi.shape)
			df_dx[i]=(fun(xi)-fun(xm1))/dx
		xip1=xi-LR*df_dx #STEP 

		if(t%10==0):
			df=np.mean(np.absolute(fun(xip1)-fun(xi)))
			
			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break
		
		xi = xip1
		t=t+1
		
		if(t%len(index)==0):
			epoch += 1
			epochs.append(epoch)

	return xi




po=np.random.uniform(0.1,1.,size=NFIT)
# print(old_loss(po))
print(new_loss(po))


# exit()
# res = minimize(new_loss, po)	
# popt = res.x

popt = my_minimizer(new_loss, po)



# exit()
# print(popt)
# print(res)

# old_loss(po)

# exit()

def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 


xm=np.array(sorted(xt))
ypred=np.array(model(xm,popt))

# print(loss_list)

print(epoch)
print(epochs)

# exit()


if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(xt, yt, 'o', label='Training set')
	ax.plot(xv, yv, 'x', label='Validation set')
	ax.plot(xtest, ytest, 'x', label = 'Test set')
	ax.plot(xm, ypred, '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()


# exit()

#FUNCTION PLOTS
# if(IPLOT):
# 	fig, ax = plt.subplots()
# 	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
# 	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
# 	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
# 	plt.xlabel('x', fontsize=18)
# 	plt.ylabel('y', fontsize=18)
# 	plt.legend()
# 	plt.show()

if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	# ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

# train_loss = []
# print(epoch)
# print("\n")
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(epochs, model(xt,popt)- yt, 'o', label='Training loss')
	# ax.plot(epochs, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer epochs', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()


exit()

# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING



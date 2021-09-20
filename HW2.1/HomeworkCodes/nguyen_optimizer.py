
import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt
import json 
from   scipy.optimize import minimize

#PARAM
# xmin=-50; xmax=50;  
# NDIM=5
# xi=np.random.uniform(xmin,xmax,NDIM) #INITIAL GUESS FOR OPTIMIZEER							
# if(NDIM==2): xi=np.array([-2,-2])
INPUT_FILE='../LectureCodes/weight.json'
DATA_KEYS=['x','is_adult','y']
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

#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
# XMEAN=np.mean(x); XSTD=np.std(x)
# YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
# x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

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




 

print("#--------GRADIENT DECENT--------")

# exit()
# epoch=0; epochs=[]; loss_list = [];

PARADIGM = 'batch'
# global xbatch, ybatch

# if (PARADIGM == 'batch'):
# 	xbatch = xt
# 	ybatch = yt

# if (PARADIGM == 'stochastic'):
# 	if (epoch <len(xt) ):
# 		i = epoch
# 			# i = i+1
# 		epoch += 1
# 		print(i, epoch)
# 		if (epoch == len(xt)):
# 			i = 0
# 			epoch +=1



# 	xbatch = xt[i]; print(i, epoch, len(xt), xbatch)
# 	ybatch = yt[i]; #print(ybatch)
# global xbatch, ybatch

def new_loss(p,index=None):
	# global epochs,epoch, loss_list
	# global xbatch, ybatch
	# global xbatch, ybatch
	
	# if (PARADIGM == 'batch'):
	# 	xbatch = xt
	# 	ybatch = yt

	# if (PARADIGM == 'stochastic'):
	# 	index = 0
	# 	if (t == 0):
	# 		index += 1
	# 		# print(i, epoch)
	# 		if (t == len(xt)):
	# 			epoch +=1
	# 			index = 0



	# 	xbatch = xt[index];# print(i, epoch, len(xt), xbatch)
	# 	ybatch = yt[index]; #print(ybatch)

		# i = i+1
		# print(i)
	xbatch = xt[index]
	ybatch = yt[index]

	#TRAINING LOSS
	yp = model(xbatch,p)
	# yp=model(xt,p) #model predictions for given parameterization p
	# training_loss=(np.mean((yp-yt)**2.0))  #MSE
	mse=(np.mean((yp-ybatch)**2.0))  #MSE

	# 
	# #RECORD FOR PLOTING
	# loss_list.append(mse)
	# epochs.append(epoch); epoch+=1


	return mse	
''

#LOSS FUNCTION
def old_loss(p):
	global epochs,loss_train,loss_val,epoch

	#TRAINING LOSS
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	yp=model(xv,p) #model predictions for given parameterization p
	validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	# if(epoch==0):    print("epoch	training_loss	validation_loss") 
	# if(epoch%25==0): print(epoch,"	",training_loss,"	",validation_loss) 
	# 
	#RECORD FOR PLOTING
	loss_train.append(training_loss); loss_val.append(validation_loss)
	epochs.append(epoch); epoch+=1

	return training_loss

#PARAM

epoch = 0
epochs = []

def my_minimizer(fun, x0):
	global epoch, epochs

	dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
	LR=0.001							#LEARNING RATE
	t=0 	 							#INITIAL epoch COUNTER
	tmax=1000							#MAX NUMBER OF epoch
	tol=10**-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	xi = x0
	NDIM = len(x0)
	print("INITAL GUESS: ",xi)

	if (PARADIGM == 'batch'):
		index = len(xt)
		# epoch += 1
		# xbatch = xt
		# ybatch = yt

	if (PARADIGM == 'stochastic'):
		index = 0
		if (t == 0):
			index += 1
			# print(i, epoch)
			if (t == len(xt)):
				# epoch +=1
				index = 0
	
	# if (PARADIGM == 'minibatch'):
	# 	mini_index = np.random.permutation(xt.shape[0])
	# 	if (t%2==0):
	# 		rand_indices = np.random.permutation(x.shape[0]) = 




		# xbatch = xt[index];# print(i, epoch, len(xt), xbatch)
		# ybatch = yt[index]; #print(ybatch)

	if (PARADIGM == 'stochastic'):
		tmax = 250
		LR =0.02
		ICLIP = True

	
	while(t<=tmax):
		

		#NUMERICALLY COMPUTE GRADIENT 
		df_dx=np.zeros(NDIM)
		for i in range(0,NDIM):
			dX=np.zeros(NDIM);
			dX[i]=dx; 
			xm1=xi-dX; print(xi,xm1,dX,dX.shape,xi.shape)
			# print(dX)
			df_dx[i]=(fun(xi)-fun(xm1))/dx
			# print(xm1)
		#print(xi.shape,df_dx.shape)
		xip1=xi-LR*df_dx #STEP 
		# xi = xip1

		if(t%10==0):
			df=np.mean(np.absolute(fun(xip1)-fun(xi)))
			# print(df)
			# print(t,"	",xi,"	","	",f(xi)) #,df) 
			
			if(df<tol):
				# print("STOPPING CRITERION MET (STOPPING TRAINING)")
				# print(df)
				break
			
		
		#UPDATE FOR NEXT epoch OF LOOP
		# xi=xip1
		epoch = epoch+1
		epochs.append(epoch)
		
		
		xi = xip1
		t=t+1

		

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
print(epochs)
exit()
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



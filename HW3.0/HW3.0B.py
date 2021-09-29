import numpy as np


# ---------------------------------------------- #
# MISC PARAMETERS
IPLOT = True
I_NORMALIZE = True
model_type = 'linear'
model_type = 'ANN'

# OPTIMIZATION PARAMS
PARADIGM = 'minibatch'
algo = 'MOM'
LR = 0.1
dx = 0.0001
max_iter = 5000
tol = 10**-10
max_rand_wb = 1.0
GAMMA_L1 = 0.0
GAMMA_L2 = 0.0001
alpha = 0.25

# ANN PARAMS
layers = [-1,2,3,2,1]
# layers = [1,5,5,5,1]
activation = "TANH"
# activation = "SIGMOID"

# ---------------------------------------------- #

# CALCULATE NUMBER OF FITTING PARAMETERS FOR SPECIFIED NN
NFIT = 0
for i in range(1, len(layers)):
    print("Nodes in layer-",i-1, " = ", layers[i-1])
    NFIT = NFIT+layers[i-1]*layers[i]+layers[i]

print("Nodes in layer-",i, " = ", layers[i])    
print("NFIT: ", NFIT)    

# RANDOM INITIAL GUESS
po = np.random.uniform(-max_rand_wb, max_rand_wb, size = NFIT)
# print(po)
print("po shape: ", po.shape)


# print(po[0:0+5])
#TAKES A LONG VECTOR W OF WEIGHTS AND BIAS AND RETURNS 
#WEIGHT AND BIAS SUBMATRICES
# def extract_submatrices(WB):
#     submatrices=[]; K=0; k0 = 0
#     for i in range(1,len(layers)-1):
#         print("i: ",i,"k0: ",k0,"Initial K: ",K)
#         #FORM RELEVANT SUB MATRIX FOR LAYER-N
#         Nrow=layers[i]
#         Ncol=layers[i-1]; 
#         # print(Ncol)
#         w=np.array(WB[k0:K+Nrow*Ncol].reshape(Ncol,Nrow).T)
#         # print(WB[K:K+Nrow*Ncol])
#         print(WB[K:K+Nrow*Ncol].shape)
#         print(Ncol, Nrow)
        
#         print(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow))
#         # Ncol = -1, Nrow = 9
#         # Because Ncol is -1, it is inferred from 18/9 = 2

#         print(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow).shape)
#         # print("w shape: ", w.shape)
#         # exit() #unpack/ W 

#         K=k0+Nrow*Ncol 
#         print("i: ",i,"k0: ",k0,"Next K: ",K)
        
#         Nrow=layers[i+1]; Ncol=1
#         b=np.transpose(np.array([WB[k0:K+Nrow*Ncol]])) #unpack/ B 
        
#         # Update K for next pass
#         K=k0+Nrow*Ncol
#         print("i: ",i,"k0: ",k0,"Updated K: ",K)

#         submatrices.append(w); submatrices.append(b)
#         print("i: ",i, "w: ", w.shape,"b: ",b.shape)
#     # print(submatrices)
#     return submatrices 
def extract_submatrices(WB):
    submatrices=[]; K=0; k0=0
    for i in range(1, len(layers)-1):
        print(i,k0,K)
        #FORM RELEVANT SUB MATRIX FOR LAYER-N
        Nrow=layers[i+1]; Ncol=layers[i] #+1
        w=np.array(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
        k0 = K
        K=K+Nrow*Ncol; print(i,k0,K)
        Nrow=layers[i+1]; Ncol=1; #+1
        b=np.transpose(np.array([WB[K:K+Nrow*Ncol]])) #unpack/ W 
        k0 = K
        K=K+Nrow*Ncol; print(i,k0,K)
        submatrices.append(w); submatrices.append(b)
        print(i, w.shape,b.shape)
    print(submatrices)
    return submatrices

test = np.array([111,112,113,211,212,213,11,12,13,121,122,212,222,321,322,21,22,131,231,33])
print(len(test))
wb = extract_submatrices(test)
print(wb)
# print(len(wb))
for val, i in enumerate(range(len(wb))):
    print(val, wb[i].shape, wb[i])
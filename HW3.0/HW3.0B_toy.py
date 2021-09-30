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



exit()
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
# print(wb)
# print(len(wb))
for val, i in enumerate(range(len(wb))):
    print(val, wb[i].shape)
# Next step: use matmul to loop over layers
# SIGMOID
# Ignore multiple inputs for now

# for index, i in enumerate(wb):
#     print(index, i)

x = np.array([1,2,3,4]).reshape(2,-1)
print(x.shape)

w = wb[0]
b = wb[1]

# print(w, w.shape)
# print(b, b.shape)

# a = np.matmul(x,wb)
print("START HERE")
# # a =np.matmul(x, w)+b
# def my_function(x0):
#     for i in range(len(wb)):
#         p_row = wb[i].shape[0]
#         p_col = wb[i].shape[1]
#         x_row = x0.shape[0]
#         x_col = x0.shape[1]

#         if p_col != x_row:
#             x0.reshape(-1,p_col)
#             print(i, wb[i].shape, x0.shape)
        
#         print(i, wb[i].shape, x0.shape)
#         mat = np.matmul(x0, wb[i])
        # x = x0.reshape()
        #print(index, "p_row: ", p_col, "p_col: ", p_row, "x_row: ", x_row,"x_col: ",x_col)
        # if shape_p[]
# def my_function(x0):
#     n_layer = 3
#     for i in range(len(wb)):
#         print(wb[i])
#         mat = np.matmul(x, wb[i])
#         print("mat: ", mat, mat.shape)
#         print(wb[i+1])
#         print(wb[i+1].shape)
#         x_new = mat + wb[i+1].reshape(-1,mat.shape[0])
#         print(x_new)
#     return(x_new)
# my_function(x)

print(x, "x shape: ", x.shape) # 2 x 2
print(w, "w shape: ", w.shape) # 3 x 2
print(b, "b shape: ",b.shape) # 3 x 1
x1 = np.array([1,2,3,4]).reshape(2,-1)
print("x1: ",x1.shape)

# def my_function(x0,w):
#     if x0.shape[1] == 1:
#         mat = np.matmul(w,x0)
#     if x0.shape[1] > 1:
#         mat = np.matmul(w, x0)
#     return mat

# z = my_function(x,w)
# print("x ", x.shape,"w: ", w.shape, "z: ",z.shape)
# y = my_function(x1,w)
# print("x1 ", x1.shape,"w: ", w.shape,"y: ",y.shape)
# if model_type == "logistic":
#     Y = S(Y)
def my_function2(x0, w):
    print("w original: ", w.shape)
    print("x original: ", x0.shape)
    w_res = w
    if x0.shape[1] != w.shape[1]:
        w_res = w.reshape(-1,x0.shape[1])
    print("w after: ", w_res.shape)
    return w_res

test = my_function2(x1, w)

print(test)
print(test.shape)

print("LOOOOOOK")

n_columns = 2
x_input = np.array([1,2,3,4]).reshape(-1,n_columns)
print(x_input.shape)



def my_function4(x0,w):
    w_reshape = w.reshape(n_columns,-1)
    mat = np.matmul(x0,w_reshape)
    return mat


print("LOOK HERE")
matrix = my_function4(x_input,w)   
print(matrix) 
print(matrix.shape)


def my_function5(x0,w,b):
    print("x :",x0.shape)
    print("w ori: ",w.shape)
    w_reshape = w.reshape(n_columns,-1)
    print("w res: ",w_reshape.shape)
    mat = np.matmul(x0,w_reshape)
    print("mat: ", mat.shape)
    print("b: ", b.shape)
    mat_b = mat +b

    print("mat_b: ", mat_b.shape)
    return mat_b

print("LOOOOOOOOOOOOK")
for i in range(len(wb)):
    print(wb[i])
    print(wb[i].shape)

nrow = 2
xs = np.array([100,200]).reshape(nrow,-1)
print("xs: ",xs.shape)

xm = np.array([100,200,1000,2000]).reshape(nrow,-1)
print("xm: ",xm.shape)

# hidden layer 1
def func(x0, p):
    o1 = None
    x = x0
    for i in [0,2,4]:
        if (x.shape[1] == 1):
            print("x0: ",x.shape)
            print("w: ",p[i].shape)
            print("b: ",p[i+1].shape)
            o1 = np.matmul(p[i], x) + p[i+1]
            print("product: ",np.matmul(p[i],x).shape)
            x = o1
            print("x updated: ",x.shape)
        if (x.shape[1] > 1 ):
            # print("x0: ",x.shape)
            # print("w_T: ",p[i].reshape(x.shape[1],-1))
            o1 = np.matmul(x,p[i].reshape(x.shape[1],-1)) + p[i+1].reshape(-1,x.shape[1])
    return o1
# func(xs,wb)

out = func(xs,wb)
print(out)
# # ------------------------
# # MODEL
# # ------------------------


# def model(x, p):
#     linear = p[0] + np.matmul(x, p[1:].reshape(NFIT - 1, 1))
#     if model_type == "linear":
#         return linear
#     if model_type == "logistic":
#         return S(linear)

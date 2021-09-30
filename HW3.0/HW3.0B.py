import numpy as np
import json

# ---------------------------------------------- #
#                   USER INPUTS                  #
# ---------------------------------------------- #

# -------------- PARAMS SELECTIONS ------------- #

# Plot and normalize
IPLOT = True
I_NORMALIZE = True

# Paradigms and algorithms 
PARADIGM = 'minibatch'
algo = 'MOM'

# Hyperparameters for optimizer tuning
LR = 0.1
dx = 0.0001
max_iter = 5000
tol = 10**-10
max_rand_wb = 1.0
GAMMA_L1 = 0.0
GAMMA_L2 = 0.0001
alpha = 0.25

# ------------- ACTIVATION FUNCTION ------------ #

activation = 'SIGMOID'          # CHOOSE FOR LOGISTIC OR ANN
# activation = 'TANH'           # CHOOSE FOR LOGISTIC OR ANN
# activation = None             # CHOOSE FOR LINEAR REGRESSION

# -------------- SELECT INPUT FILE ------------- #

INPUT_FILE = "./planar_x1_x2_y.json"
# INPUT_FILE = "./planar_x1_x2_x3_y.json"

with open(INPUT_FILE) as f:
    my_input = json.load(f)  # read into dictionary

# ---------------- DEFINE X & Y ---------------- #

# Read x & y based on input file
if (INPUT_FILE == "./planar_x1_x2_y.json"):
    X_KEYS = ["x1","x2"]
    Y_KEYS = ["y"]

if (INPUT_FILE == "./planar_x1_x2_x3_y.json"):
    X_KEYS = ["x1","x2","x3"]
    Y_KEYS = ["y"]

# --------------- LAYER DEFINITION -------------- #

layers = [-1,2,3,2,1]                     # CHOOSE FOR ANN
# layers = [len(X_KEYS), len(Y_KEYS)]         # CHOOSE FOR LINEAR OR LOGISTIC


# -------- NUMBER OF FITTING PARAMETERS -------- #

NFIT = 0
for i in range(1, len(layers)):
    NFIT = NFIT+layers[i-1]*layers[i]+layers[i]

# ---------------------------------------------- #
#                   MODEL TYPE                   #
# ---------------------------------------------- #

# Based on user input 
if (activation == None):
    model_type = "linear"

if (activation == 'SIGMOID') or (activation == 'TANH'):
    if (NFIT > 4):
        model_type = 'ann'
    if (NFIT <= 4):
        model_type = 'logistic'

# ------------- CONFIRM MODEL TYPE ------------- #

print("CONFIRMING MODEL TYPE BASED ON USER INPUTS:", model_type)
print("WITH ", NFIT, "FITTING PARAMETERS")
print("ACTIVATION FUNCTION IS ", activation)

if (model_type == "ann"):
    for i in range(1, len(layers)):
        print("Nodes in layer-",i-1, " = ", layers[i-1])
    # NFIT = NFIT+layers[i-1]*layers[i]+layers[i]

    print("Nodes in layer-",i, " = ", layers[i])    
    print("NFIT: ", NFIT)    


# ---------------------------------------------- #
#                 DEFINE FUNCTION                #
# ---------------------------------------------- #

def extract_submatrices(WB):
    submatrices=[]; K=0; k0=0
    for i in range(1, len(layers)-1):
        # print(i,k0,K)
        #FORM RELEVANT SUB MATRIX FOR LAYER-N
        Nrow=layers[i+1]; Ncol=layers[i] #+1
        w=np.array(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
        k0 = K
        K=K+Nrow*Ncol
        # print(i,k0,K)
        Nrow=layers[i+1]; Ncol=1; #+1
        b=np.transpose(np.array([WB[K:K+Nrow*Ncol]])) #unpack/ W 
        k0 = K
        K=K+Nrow*Ncol; print(i,k0,K)
        submatrices.append(w); submatrices.append(b)
        # print(i, w.shape,b.shape)
    # print(submatrices)
    return submatrices

# ---------------------------------------------- #
#                 DO THINGS HERE                 #
# ---------------------------------------------- #

long_vector = np.array([111,112,113,211,212,213,11,12,13,121,122,212,222,321,322,21,22,131,231,33])

wb_mat = extract_submatrices(long_vector)

# ---------------------------------------------- #
#                    TOY CODE                    #
# ---------------------------------------------- #

# RANDOM INITIAL GUESS FOR FITTING PARAMETERS
po = np.random.uniform(0.1, 1.0, size=NFIT)

# EXTRACT WEIGHT & BIAS VECTORS
my_p = extract_submatrices(po)

long_vector = np.array([111,112,113,211,212,213,11,12,13,121,122,212,222,321,322,21,22,131,231,33])

wb = extract_submatrices(long_vector)

#weights & biases matrix
for layer, i in enumerate(range(len(wb))):
    print("Layer: ", layer, " --- shape: ", wb[i].shape)
    print(wb[i], "\n")


# ---------------------------------------------- #
#                READ IN REAL DATA               #
# ---------------------------------------------- #

# MANIPULATE DATA

X = []
Y = []

for key in my_input.keys():
    if key in X_KEYS:
        X.append(my_input[key])
    if key in Y_KEYS:
        Y.append(my_input[key])


# MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X = np.transpose(np.array(X))
Y = np.transpose(np.array(Y))


print("--------INPUT INFO-----------")
print("X shape:", X.shape)
print("Y shape:", Y.shape, "\n")

# ---------------------------------------------- #
#             MANUAL CALCULATION TEST            #
# ---------------------------------------------- #
print("LOOK HERE")


print(X.shape)
print(wb[0].shape)
print(wb[1].shape)

print("KEEP LOOKING")
# h1_out = np.matmul(X,wb[0]) + wb[1]
h1_out = np.matmul(X, wb[0].T + wb[1].T)
print(h1_out.shape)

print(h1_out.shape, wb[2].shape, wb[3].shape)
h2_out = np.matmul(h1_out, wb[2].T) +wb[3].T

print(h2_out.shape, wb[4].shape, wb[5].shape)
nn_out = np.matmul(h2_out, wb[4].T) +wb[5].T

print(nn_out.shape, nn_out)
print("END PSEUDO CODE")

# ------------ss---------------------------------- #
hidden_layers = layers[1:-1]
print(hidden_layers)
print(len(hidden_layers))

print("LOOK HERE NOW")
print(list(range(len(hidden_layers))))

loop = list(range(len(hidden_layers)))

# ---------------------------------------------- #
print(loop)
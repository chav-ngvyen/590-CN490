import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize


###################################
model_type = 'linear'
# model_type = 'logistic'
NFIT = 1
OPT_ALGO = 'Nelder-Mead'
test_size = 0.2
###################################
with open('/home/chau/590-CODES/DATA/weight.json') as f:
    data = json.load(f)

###############################
class Data():
    def __init__(self, attributions):
        self.is_adult = np.array(attributions['is_adult'])  
        # x is age  
        self.age = np.array(attributions['x'] )   
        # y is weight
        self.weight= np.array(attributions['y'])

################################
# Linear regression adapted from this https://towardsdatascience.com/linear-regression-from-scratch-cd0dee067f72
def params(x, p):
    x_mean = np.mean(x)
    p_mean = np.mean(p)

    cov_x_p = 0
    var_x = 0 

    for i in range(len(x)):
        cov_x_p += (x[i] - x_mean)*(p[i] - p_mean)
        var_x += (x[i] - x_mean)**2
    
    #m is slope
    m = cov_x_p / var_x
    #b is bias
    b = p_mean - (m*x_mean)
    return m, b

# train test split
def split(input_array):
    global test_size 
    size = test_size
    train, test = train_test_split(input_array, test_size = size, random_state = 42)
    return np.asarray(train), np.asarray(test)

# Normalize function
def normalize(original_array):
    normalized_array = (original_array - np.mean(original_array)) / np.std(original_array)
    return normalized_array

# Reverse scaling
def reverse_normalize(normalized_array, original_mean, original_std):
    reversed_array = original_std *normalized_array + original_mean
    return reversed_array 

# Linear prediction
def my_predict(test_set,m,b):
    if model_type == "linear":
        y_pred = b + m * test_set   
    if model_type == "logistic":
        pass
    return(y_pred)

# Loss function

def loss(p):
    global y_predict
    loss = np.zeros_like(y_predict)
    ### https://www.kdnuggets.com/2019/03/neural-networks-numpy-absolute-beginners-part-2-linear-regression.html/2
    loss = 1/2 * np.mean((y_predict - p)**2)
    return loss
################################
# Initialize
my_data = Data(data)

# Train test split
x_train, x_test = split(my_data.age)
y_train, y_test = split(my_data.weight)

# Normalize
x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)
y_train_norm = normalize(y_train)

# Make predictions
m, b = params(x_train_norm, y_train_norm)
y_pred_norm = my_predict(x_test_norm, m, b)

# Rescale
y_predict = reverse_normalize(y_pred_norm, np.mean(y_train), np.std(y_train) )

print(m)
print(b)

##################################
# Plot

fig, ax = plt.subplots()
ax.plot(x_train, y_train, '.')
xe = np.linspace(0,18,100)
ye = xe*m + b
ax.plot(xe,ye)
# plt.show()
# exit()

#################################
po=np.random.uniform(0.5,1., size = NFIT)
print(type(po))
print(po)
res = minimize(loss, po, method=OPT_ALGO, tol=1e-15)
popt = res.x
print("OPTIMAL PARAM:",popt)
plt.show()
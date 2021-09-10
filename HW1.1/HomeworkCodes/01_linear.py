import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
        
    

my_data = Data(data)
print(type(my_data.is_adult))

# not_adult = [val for val in my_data if my_data['is_adult'] == 0]
# print(not_adult)
# print(len(not_adult))

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

def normalize(original_array):
    normalized_array = (original_array - np.mean(original_array)) / np.std(original_array)
    return normalized_array

def reverse_normalize(normalized_array, original_mean, original_std):
    reversed_array = original_std *normalized_array + original_mean
    return reversed_array 

def my_predict(test_set,m,b):
    y_pred = b + m * test_set
    return(y_pred)

################################
# Initialize
my_data = Data(data)
print(my_data)
my_data = my_data[my_data.age<18]
# Train test split
x_train, x_test = split(my_data.age)

y_train, y_test = split(my_data.weight)

# Normalize
x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)
# x_test_norm = normalize(x_test[x_test<18])
y_train_norm = normalize(y_train)

# Make predictions
m, b = params(x_train_norm, y_train_norm)
y_pred_norm = my_predict(x_test_norm, m, b)

# Rescale
y_pred = reverse_normalize(y_pred_norm, np.mean(y_train), np.std(y_train) )

print(m)
print(b)
##################################
# Plot
# Plot
fig, ax = plt.subplots()
# m_test, b_test = np.polyfit(x_train[x_train<18], y_train,1)
# plt.plot(x_train, m*x_train + b)
# ax.plot(x_test, y_test, 'o')
ax.plot(x_train, y_train, '.')
xe = np.linspace(0,18,100)
ye = xe*m + b
ax.plot(xe,ye)
plt.show()
exit()
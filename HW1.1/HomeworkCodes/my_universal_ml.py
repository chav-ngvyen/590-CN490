import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt

###################################
model_type = 'linear'
# model_type = 'logistic'
NFIT = 1
OPT_ALGO = 'Nelder-Mead'
test_size = 0.2
###################################
with open('/home/chau/590-CODES/DATA/weight.json') as f:
    data = json.load(f)
    # print(type(data))
    # print(dir(data))
    # print(data.items())
    # print(data.keys())

# exit()
from sklearn.model_selection import train_test_split
class Data():
    def __init__(self, attributions):
        self.is_adult = attributions['is_adult']   
        # x is age  
        self.age = attributions['x']    
        # y is weight
        self.weight= attributions['y']


my_data = Data(data)



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

def split(input_array):
    global test_size 
    size = test_size
    train, test = train_test_split(input_array, test_size = size, random_state = 42)
    return np.asarray(train), np.asarray(test)

# exit()    
class scaler():
    def __init__(self, input):
        self.input = input
        self.mean = np.mean(self.input)
        self.std = np.std(self.input)
    
    # This is so I can print an instance of the object out
    def __str__(self):
        return f'{self.input}'

    def normalize(self):
        self.norm = (self.input - self.mean) / self.std
        return self.norm

    def inverse(self):
        return self.std*self.normalize() + self.mean

my_scaler = scaler([1,2,3])

norm = my_scaler.normalize()
print(norm)
print("\n")
print(type(norm))
print("\n")
print(dir(norm))
# ### Used this to test my_array
# arr = my_array([1,2,3,4,5,6,7,8])
# print(arr)
# # print(dir(arr))
# # print(arr.mean)
# # print(arr.std)
# # print(arr.normalize())
# # print(arr.inverse())
# # exit() 
#       
def normalize(original_array):
    normalized_array = (original_array - np.mean(original_array)) / np.std(original_array)
    return normalized_array, np.mean(original_array), np.std(original_array)

def reverse_normalize(normalized_array, original_mean, original_std):
    reversed_array = original_std *normalized_array + original_mean
    return reversed_array    


# #############################################################
# # Used this to test normalize & reverse normalize functions
# X_train = np.array([ 1,2,3,4,5,6,7,8])
# # print(normalize(X_train))
# # print(reverse_normalize(X_train, normalize(X_train)))
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_transformed = scaler.fit_transform(X_train[:, np.newaxis])
# print(X_transformed)
# print(scaler.inverse_transform(X_transformed))
# exit()    

# def model(x, p):
#     global model_type
#     m, b = params(x, p)
#     if model_type == "linear":
#         y_pred = b + np.multiply(m, x) 
#     if model_type == "logistic":
#         pass 
#     return y_pred

def my_predict(test_set,m,b):
    y_pred = b + m * test_set
    return(y_pred)
# Loss function to optimize (minimize)
def loss(p):
    global y_predict
    loss = np.zeros_like(y_predict)
    ### https://www.kdnuggets.com/2019/03/neural-networks-numpy-absolute-beginners-part-2-linear-regression.html/2
    loss = 1/2 * np.mean((y_predict - p)**2)
    return loss

## From https://www.askpython.com/python/examples/linear-regression-from-scratch
# class LinearRegression:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.m = 0
#         self.b = 0
#         self.n = len(x)
    
    # def fit(self , epochs , lr):
    #     ## epoch is number of iterations
    #     ## l is learning rate
        
    #     ##Implementing Gradient Descent
    #     for i in range(epochs):
    #         y_pred = self.m * self.x + self.b
             
    #         #Calculating derivatives w.r.t Parameters
    #         D_m = (-2/self.n)*sum(self.x * (self.y - y_pred))
    #         D_b = (-1/self.n)*sum(self.x-y_pred)
             
    #         #Updating Parameters
    #         self.m = self.m - lr * D_m
    #         self.b = self.b - lr * D_b
             
    # def predict(self , input):
    #     y_pred = self.m * input + self.b 
    #     return y_pred
    
# exit()    
#################################
my_data = Data(data)

# exit()
x_train, x_test = split(my_data.age)
y_train, y_test = split(my_data.weight)

print(x_train.shape)
print(y_train.shape)

# exit()
from sklearn.linear_model import LinearRegression
print("check here")
# reg = LinearRegression().fit(x_train.reshape(-1,1),y_train)

# print(reg.coef_, reg.intercept_)

print("check again")


# print("my predict")
# print(my_predict(x_test)) #y_pred

# print("sklearn predict")
# print(reg.predict(x_test.reshape(-1,1))) #y_pred
###################################


x_train_norm, x_train_mean, x_train_std = normalize(x_train)
x_test_norm, x_test_mean, x_test_std = normalize(x_test)

y_train_norm, y_train_mean, y_train_std = normalize(y_train)


m, b = params(x_train_norm, y_train_norm)
my_y_scaled = my_predict(x_test_norm, m, b)
# print(m,b)
# exit()
print(my_y_scaled)

# print(y_train_std)

# exit()
y_reverse = reverse_normalize(my_y_scaled, y_train_mean, y_train_std)

# print(y_reverse)
# exit()
# exit()

###################################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x_train.reshape(-1,1))
# print(x_scaled)
y_scaled = scaler.fit_transform(y_train.reshape(-1,1))
# print(y_scaled)
reg = LinearRegression().fit(x_scaled, y_scaled)
# print(reg.coef_)
# print(reg.intercept_)

print("what")
y_pred_scaled = reg.predict(scaler.fit_transform(x_test.reshape(-1,1)))
print(y_pred_scaled)
print(reg.coef_)
print(reg.intercept_)
exit()
y_pred_trans = scaler.inverse_transform(y_pred_scaled)
print(y_pred_trans)
exit()
# x_scale_reverse = scaler().inverse_transform()
# print("x train norm \n")
# print(x_train_norm)
# print("x scaled \n")
# print(x_scaled)

print(x_reverse)

exit()
################################
print(scaler.inverse_transform(x_scaled))
print(x_train.reshape(1,-1))

exit()
print("scaler mean")
print(scaler.mean_)
print("scaler scale")
print(scaler.scale_)

exit()
x_train = my_array(x_train)
y_train = my_array(y_train)

x_train_norm = x_train.normalize()
y_train_norm = y_train.normalize()

# regressor = LinearRegression(x_train_norm, y_train_norm)
# regressor.fit(1000,0.01)
y_pred_norm = model(x_train_norm,y_train_norm)
y_pred1 = y_pred_norm*y_train.std+y_train.mean

print(y_pred_norm, y_pred_norm.shape)
print(y_pred1, y_pred1.shape)
# # exit()
# # print(regressor)
# y_pred_norm = regressor.predict(x_test)
# print(y_pred_norm)

# exit()
# print(type(y_pred_norm))
# y_pred_norm1 = my_array(regressor.predict(x_test_norm))
# print(y_pred_norm1)
# # print(type(y_pred_norm1))

# y_pred1 = y_pred_norm*y_train.std+y_train.mean
# print(y_pred1)
# print(x_test)
# exit()
############################
# Plot
fig, ax = plt.subplots()
# ax.plot(x_train_norm, y_train_norm, 'o', label = "Training set")
# ax.plot(x_test_norm, y_pred_norm, 'x', label = 'Predict')
# ax.plot(x_train, y_train, 'o', label = "Training set")
ax.plot(x_test, y_pred1, 'x', label = 'Predict')

plt.show()
exit()
# print(linear_reg(my_data.age, my_data.weight))


#################################
from scipy.optimize import minimize

po=np.random.uniform(0.5,1., size = NFIT)
print(type(po))
print(po)
res = minimize(loss, po, method=OPT_ALGO, tol=1e-15)
popt = res.x
print("OPTIMAL PARAM:",popt)

exit()
# #######
# Compare to linear regression from sklearn
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.asarray(my_data.age).reshape(-1,1), np.asarray(my_data.weight))

print(reg.intercept_)
print(reg.coef_)

#exit()
#######

# def logistic_reg(x, p):
#     x_mean = np.mean(x)
#     p_mean = np.mean(p)

#     cov_x_p = 0
#     var_x = 0 

#     for i in range(len(x)):
#         cov_x_p += (x[i] - x_mean)*(p[i] - p_mean)
#         var_x += (x[i] - x_mean)**2
    
#     #m is slope
#     m = cov_x_p / var_x
#     #b is bias
#     b = p_mean - (m*x_mean)

#     # predicted value of p
#     p_pred = p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))
#     return p_pred


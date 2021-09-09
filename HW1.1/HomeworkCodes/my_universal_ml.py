import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt


with open('/home/chau/590-CODES/DATA/weight.json') as f:
    data = json.load(f)
    print(type(data))
    print(dir(data))
    #print(data.items)
    #print(data.keys())


class Data():
    def __init__(self, attributions):
        self.is_adult = attributions['is_adult']     
        self.age = attributions['x']    
        self.wage= attributions['y']



# class Data():
#     def __init__(self):
#         with open(self) as f:
#             return json.load(f)    


my_data = Data(data)
print(np.average(my_data.age))


def linear_reg(x, p):
    # m is slope (rise/ run)
    m = np.sum((x-np.average(x))*(p-np.average(p)))
    # b is constant
    return m

print(linear_reg(my_data.age, my_data.wage))

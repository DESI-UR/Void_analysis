#Debra Dunham
#7/13/2020

import pickle
import numpy as np

temp_file = open('void_dictionary.pickle', 'rb')
dictionary = pickle.load(temp_file)
temp_file.close()

x = np.zeros(len(dictionary))
y = np.zeros(len(dictionary))
z = np.zeros(len(dictionary))
r = np.zeros(len(dictionary))

scale_dict = {}

i = 0

for a in dictionary:
    print('working with key ', a)
    for b in range(len(dictionary[a])):
        if dictionary[a][b][0] == 1.0:
            x[i] = dictionary[a][b][1]
            y[i] = dictionary[a][b][2]
            z[i] = dictionary[a][b][3]
            r[i] = dictionary[a][b][4]
    for b in range(len(dictionary[a])):
        if x[i]!=0.0:
            value = dictionary[a][b][0]
            e = x[i] - dictionary[a][b][1]
            f = y[i] - dictionary[a][b][2]
            g = z[i] - dictionary[a][b][3]
            d = np.sqrt(e**2 + f**2 + g**2)
            R = dictionary[a][b][4] / r[i]
            scale_dict.setdefault(a,[]).append([value, d, R])
    i = i + 1
#print(scale_dict)  

temp_file = open('scaled_void_dictionary.pickle', 'wb')
pickle.dump(scale_dict, temp_file)
temp_file.close()
print('created   scaled_void_dictionary.pickle   file')

print(len(dictionary))
print(len(scale_dict))
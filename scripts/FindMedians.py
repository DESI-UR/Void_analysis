from time import time
import pickle
import numpy as np
from astropy.table import Table

T = time()

temp_file = open('hole_table.pickle', 'rb')
table = pickle.load(temp_file)
temp_file.close()

file_numbers = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
medians = np.zeros(len(file_numbers))

for a in range(len(file_numbers)):
    t = time()
    distances = np.zeros(1)
    for b in range(1, len(table['dist'])):
        if table['file'][b]==file_numbers[a]:
            distances = np.append(distances, table['dist'][b])
    medians[a] = np.median(distances)
    print('for iteration of ', file_numbers[a], 'the median is ', medians[a], 'and it took ', time()-t)

temp_file = open('medians.pickle', 'wb')
pickle.dump(medians, temp_file)
temp_file.close()
print('total time: ', time()-T)
print(medians)
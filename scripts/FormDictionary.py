import pickle
from time import time

t = time()

begin = '/scratch/ddunham7/Voids/HoleShift/VF_DEBUG'
#begin = 'Tuple'
number = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
#number = [1,2]
end = '.txt'

dictionary = {}

for a in range(len(number)):
    file_name = begin + str(number[a]) + end
    print('begin reading file ', file_name)
    infile = open(file_name, 'r')
    for lines in infile:
        line = lines.split()
        if line[1]=='hole':
            l = line[2].split(',')
            dictionary.setdefault(line[0], []).append([number[a], float(l[0]), float(l[1]), float(l[2]), float(l[3])])
            
    infile.close()
    print('done reading file ', file_name, ' ', time()-t)
temp_file = open('void_dictionary.pickle', 'wb')
pickle.dump(dictionary, temp_file)
temp_file.close()
print('created   void_dictionary.pickle   file')

import matplotlib.pyplot as plt
import numpy as np
import csv
from os import listdir
# x = np.loadtxt('data/new_14.txt', delimiter='\n', unpack=True)
# print (x)
# print(x.shape)
# plt.plot(x, label='Loaded from file!')

# plt.show()


#Create and save CSV

mypath ='data/csv/'
result = []
lst = listdir(mypath)
lst.sort()
print(lst)
for filename in lst:
	with open('data/csv/' + filename) as csvfile:
	    readCSV = csv.reader(csvfile, delimiter=',')
	    for row in readCSV:
	        result.append(row[1])

# with open('new.txt', 'w') as f:
#     for item in result:
#         f.write("%s\n" % item)

plt.plot(result, label='Loaded from file!')
plt.show()
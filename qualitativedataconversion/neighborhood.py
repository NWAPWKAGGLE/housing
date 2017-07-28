import numpy as np
from tqdm import tqdm

#load in qualitative data
qualitativedata = np.genfromtxt('./qualitativedata.csv', delimiter=',', dtype=None)
qualitativetestdata = np.genfromtxt('./testdata.csv', delimiter=',', dtype = None)

print(qualitativedata)

#initialize arrays
options = [[0] * 1 for i in range(len(qualitativedata[0]))]
quantitativedata = [[0] * 1 for j in range(len(qualitativedata[0]))]
quantitativetestdata = [[0] * 1 for k in range(len(qualitativetestdata[0]))]
print(quantitativedata)
print(options)

#loop through and insert values by option number
for j in tqdm(range(len(qualitativedata[0]))):
    for i in tqdm(range(len(qualitativedata))):
        try:
            if (options[j].index(qualitativedata[i][j])):
                quantitativedata[j].append(options[j].index(qualitativedata[i][j]))
        except:
            options[j].append(qualitativedata[i][j])
            print(options[j])
            quantitativedata[j].append(len(options[j])-1)

for j in tqdm(range(len(qualitativetestdata[0]))):
    for i in tqdm(range(len(qualitativetestdata))):
        try:
            if (options[j].index(qualitativetestdata[i][j])):
                        quantitativetestdata[j].append(options[j].index(qualitativetestdata[i][j]))
        except:
            options[j].append(qualitativetestdata[i][j])
            quantitativetestdata[j].append(len(options[j]) - 1)

#transpose the array
output = np.array(quantitativedata).transpose()
testoutput = np.array(quantitativetestdata).transpose()
#delete the first initialization values
output = np.delete(output, 0, axis=0)
print(output)
#save to a csv file
np.savetxt('quantitativedata.csv', output , delimiter=',', fmt='%f')
np.savetxt('quantitativetestdata.csv', output, delimiter=',', fmt='%f')
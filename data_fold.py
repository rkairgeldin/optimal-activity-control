import networkx as nx
import numpy as np
import os
import math


if __name__ == '__main__':
    path = os.getcwd()

    with open(path+"\\processed_data\\_DTDG_Graph.txt", "r") as f:
        N = int(f.readline())

    perm = np.random.permutation(N)

    f_s = math.floor(int(N)/10)
    for i in range(10):
        test = perm[i*f_s:(i+1)*f_s]
        
        
        train = np.random.permutation(list(set(perm) - set(test)))
        with open(path+"\\processed_data\\train_idx-"+str(i+1)+".txt", 'w')as f:
            for k in train:
                f.write(str(k)+'\n')
        with open(path+"\\processed_data\\test_idx-"+str(i+1)+".txt", 'w')as f:
            for k in test:
                f.write(str(k)+'\n')

DECIMAL = 2 #the resolution
prob = 0.3
S = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]#infected nodes

parameters={
    'sigma_1_I': 3, #given it is inactive it becomes active after exponential time with rate sigma_1_i
    'sigma_2_I': 1.5, #given it is active it becomes inactive after exponential time with rate sigma_2_i
    'sigma_1_H': 4, #given it is inactive it becomes active after exponential time with rate sigma_1_i
    'sigma_2_H': 2.5, #given it is active it becomes inactive after exponential time with rate sigma_2_i
    'beta_I': 3.5, #transmit I->S (susceptible becomes carrier)
    'beta_C': 2.3, #transmit C->S  (susceptible becomes carrier)
    'gamma': 1.7, #cure rate I-> R
    'nu_I': 4.5, #C->I
    'nu_R': 1.5 #C->R
}
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
file_name = sys.argv[1]
a  = np.load(file_name)
a = a['arr_0']


qw = []
for i in range(200):
    qw.append(i+1)

qw = np.asarray(qw)

if True:
    #plt.figure(1)
    #plt.subplot(221)

    plt.plot(qw,a, color='red')

    plt.ylabel('Average Reward')
    plt.xlabel('Episode number')
    plt.savefig('exp.png')


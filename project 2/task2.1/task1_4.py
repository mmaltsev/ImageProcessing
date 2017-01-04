import numpy as np
import matplotlib.pyplot as plt
import time
import task1

time_all = np.zeros([3,10])

for i in range(3):
	grid_index = 0
	for j in range(3,22,2):
		time_meas = np.zeros(10)
		for k in range(10):
			t0 = time.clock()
			task1.main(j,j,i)
			time_meas[k] = time.clock() - t0
		time_all[i][grid_index] = np.sum(time_meas)/10
		grid_index += 1
np.savetxt('time.txt', time_all, delimiter=',')
grid_size = np.arange(3,22,2)
type0_time = time_all[0][:]
type1_time = time_all[1][:]
type2_time = time_all[2][:]
plt.plot(grid_size,type0_time,'r',grid_size,type1_time,'g',grid_size,type2_time,'b')
plt.ylabel('time')
plt.xlabel('filter size')
plt.title('run-times vs filter size')
plt.grid(True)
plt.legend(['naive convolution','separable convolution','mult. in the freq. domain'])
plt.savefig("time.jpg")
plt.show()
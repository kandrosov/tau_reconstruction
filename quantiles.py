import tensorflow as tf
######## Memory allocation:
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True) # dynamic memory allocation

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]) # => uses effectively 6367 MB
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pp

# with open('quantile_evaluation_zoom.csv', newline='') as f:
#     quantiles_list = list(csv.reader(f))

# quantiles = np.array(quantiles_list)
# print(quantiles.shape) #(2,13)

# quantiles = quantiles.astype(np.float)
# q_shape= quantiles.shape
# # print(len(quantiles[0]))
# # print(int((len(quantiles[0])-2)/2))
# a = np.zeros((q_shape[0],int((len(quantiles[0])-2)/2)))
# ### Find the maximum:
# # print('test: ',np.arange(0,len(quantiles[0])-2,2))
# counter = 0
# for i in np.arange(0,len(quantiles[0])-2,2):
#     a[:,counter] = quantiles[:,i+2]-quantiles[:,i]
#     counter+=1
# print('counter: ',counter)
# # print(a)
# find_center = np.argmin(np.abs(a), axis=1)
# print(find_center) # [5,5] => 1.1 for both => indices 11,11

# find_center = find_center*2+1
# print(find_center)

# xaxis = np.arange(1,21,1)
# diff_q = np.zeros(len(xaxis))
# def_diff_q = np.zeros(len(xaxis))
# for i in xaxis:
#     # print(i)
#     # print(find_center[1]+i)
#     diff_q[i-1] = np.abs(quantiles[0,find_center[0]+i] - quantiles[0,find_center[0]-i])
#     def_diff_q[i-1] = np.abs(quantiles[1,find_center[1]+i] - quantiles[1,find_center[1]-i])

# fig0, axes = plt.subplots(1, sharex=False, figsize=(12, 8))
# fig0.suptitle('Quantiles zoom')
# plt.ylabel("Quanitile absolute difference", fontsize=14)
# plt.xlabel("Quantile interval around maximum in %", fontsize=14)
# plt.xticks(xaxis*2*0.1)
# plt.grid(True)
# plt.plot(xaxis*2*0.1, diff_q     , 'bo--', label="predicted")
# plt.plot(xaxis*2*0.1, def_diff_q , 'ro--', label="default")
# plt.legend()

# pdf0 = pp.PdfPages("../Plots/Quantiles_zoom.pdf")
# pdf0.savefig(fig0)
# pdf0.close()
# plt.close()


#####################################################################################
with open('quantile_evaluation.csv', newline='') as f:
    quantiles_list = list(csv.reader(f))

quantiles = np.array(quantiles_list)
print(quantiles.shape)

quantiles = quantiles.astype(np.float)
# find_center = np.argmin(np.abs(quantiles+0.05433), axis=1)

a = np.zeros((2,int((len(quantiles[0])-2)/2)+1))
### Find the maximum:
# print('test: ',np.arange(0,len(quantiles[0])-2,2))
counter = 0
for i in np.arange(0,len(quantiles[0])-2,2):
    # print(counter)
    # print(i)
    a[:,counter] = quantiles[:,i+2]-quantiles[:,i]
    counter+=1
# print('counter: ',counter)
# print(a)
find_center = np.argmin(np.abs(a), axis=1)
print(find_center) # [5,5] => 1.1 for both => indices 11,11

find_center = find_center*2+1
print(find_center)

xaxis = np.arange(1,31,1)
diff_q = np.zeros(len(xaxis))
def_diff_q = np.zeros(len(xaxis))
for i in xaxis:
    diff_q[i-1] = np.abs(quantiles[0,find_center[0]+i] - quantiles[0,find_center[0]-i])
    def_diff_q[i-1] = np.abs(quantiles[1,find_center[1]+i] - quantiles[1,find_center[1]-i])
    if(i==14):
        print(quantiles[0,find_center[0]+i])
        print(quantiles[0,find_center[0]-i])
        print(quantiles[1,find_center[1]+i])
        print(quantiles[1,find_center[1]-i])


fig0, axes = plt.subplots(1, sharex=False, figsize=(12, 8))
fig0.suptitle('Quantiles')
plt.ylabel("Quanitile absolute difference", fontsize=14)
plt.xlabel("Quantile interval around maximum in %", fontsize=14)
plt.xticks(xaxis*2)
plt.grid(True)
plt.plot(xaxis*2, diff_q     , 'bo--', label="predicted")
plt.plot(xaxis*2, def_diff_q , 'ro--', label="default")
plt.legend()

pdf0 = pp.PdfPages("../Plots/Quantiles.pdf")
pdf0.savefig(fig0)
pdf0.close()
plt.close()

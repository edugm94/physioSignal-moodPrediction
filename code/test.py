from utils.SignalProcessing import SignalProcessing
from utils.CreateDataset import CreateDataset
import pandas as pd
import numpy as np
from datetime import datetime
signal_temp = 'temp'
signal_acc = 'acc'
signal_hr = 'hr'
path_temp = "data/datos_E4/P1/P1_Complete/TEMP1.csv"
path_acc = "data/datos_E4/P1/P1_Complete/ACC1.csv"
path_hr = "data/datos_E4/P1/P1_Complete/HR1.csv"
path_ema = "data/datos_E4/P1/P1_Complete/EMAs1.xlsx"

temp_sp = SignalProcessing(
    type_signal=signal_temp,
    path_to_file=path_temp,
    path_to_ema=path_ema,
    window_size=60,
    type_label='happiness'
)
acc_sp = SignalProcessing(
    type_signal=signal_acc,
    path_to_file=path_acc,
    path_to_ema=path_ema,
    window_size=220,         # stress test
    type_label='happiness'
)
hr_sp = SignalProcessing(
    type_signal=signal_hr,
    path_to_file=path_hr,
    path_to_ema=path_ema,
    window_size=220,        # stress test
    type_label='happiness'
)

#acc_sp.getGroundTruth()
#arr_vector, arr_label = acc_sp.getGroundTruth()
#hr_sp.getGroundTruth()
#hr_sp.procSignal()
#df_temp = temp_sp.df
#df_hr = hr_sp.df


#print("Timestamp inicial: ",datetime.fromtimestamp(1521177894.000000))
#print("U", datetime.fromtimestamp(1521177894.000000+58679))
#print("U", datetime.fromtimestamp(1521236563.0))


dataset = CreateDataset(
    num_patients=1,
    sampling_days=[3],
    path_to_csv='data/datos_E4/',
    output_path='out',
    output_filename='test.hdf5',
    window_size=60,
    type_label='mood'
)
dataset.createDataset()
print("Hola")


'''
a = np.array([1, 2, 3])
b = np.pad(a, 4, mode='constant', constant_values=(9))
print(b)
'''

''''
# To pad an array with zeroes
an_array = np.array([[1, 2], [3, 4]])
shape = np.shape(an_array)
padded_array = np.zeros((3, 3))
padded_array[:shape[0],:shape[1]] = an_array
print(padded_array)
'''

'''
a = np.array([1,1])
b = np.array([2,2,2])
c = np.array([3,3,3])

total = []
total.append(a)
total.append(b)
total.append(c)

arr = np.asarray(total)
print(arr.shape)
'''

'''
a = np.random.randn(1, 11)
b = np.random.randn(1, 8)

print("a:\n",a)
print("b\n",b)

dif = a.shape[-1] - b.shape[-1]
i=3
if (dif%2) == 0:
    if i == 0:
        pad_width = int(dif/2)
        c = np.pad(np.squeeze(b), (0, 2*pad_width), mode='mean')
    elif i == 5:
        pad_width = int(dif / 2)
        c = np.pad(np.squeeze(b), (2*pad_width, 0), mode='mean')
    else:
        pad_width = int(dif / 2)
        c = np.pad(np.squeeze(b), (pad_width), mode='mean')

else:
    if i == 0:
        pad_width = int(dif / 2)
        c = np.pad(np.squeeze(b), (0, 2*pad_width + 1), mode='mean')
    elif i == 5:
        pad_width = int(dif / 2)
        c = np.pad(np.squeeze(b), (2 * pad_width + 1, 0), mode='mean')
    else:
        pad_width = int(dif / 2)
        c = np.pad(np.squeeze(b), (pad_width, pad_width + 1), mode='mean')

print("c:\n",c.reshape(1, -1))
'''

'''
a = np.random.rand(5, 1, 4)
print(a.shape)
if a.shape[0] != 10:
    diff = 10 - a.shape[0]
    npad = ((0, diff), (0, 0), (0, 0))
    b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)

print(b.shape)

print(b.shape[0])
'''




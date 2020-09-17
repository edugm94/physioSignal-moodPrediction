import numpy as np
import h5py

'''
with h5py.File("mytestfile.h5", "a") as f:
    arr = np.arange(10)
    dset = f.create_dataset("/MyGroup1/myDataset", (10,), dtype="i", data=arr)

'''
'''
with h5py.File("mytestfile.h5", "r") as f:
    arr = f.get('MyGroup1/myDataset')
    print(np.array(arr))

    #dset = f.get('myDataset')
    #dset = np.array(dset)
    #print(dset)
'''
##############
'''
f = h5py.File('test.h5', 'w')
arr = np.arange(10)
f.create_dataset(name="myDataset", data=arr)
f.close()
'''

'''
f = h5py.File('test.h5', 'r')
arr = np.array(f.get('myDataset'))
print(arr)
'''
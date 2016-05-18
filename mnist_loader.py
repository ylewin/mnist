# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def to_int(x):
    return int.from_bytes(x, "big")

def read_data(fpath):
    # file = gzip.open(, 'rb')
    with open(fpath, "rb") as f:
        #The magic number is an integer (MSB first). The first 2 bytes are always 0.
        for i in range(2):
            assert(to_int(f.read(1)) == 0)
        #The third byte codes the type of the data: 
        #0x08: unsigned byte 
        #0x09: signed byte 
        #0x0B: short (2 bytes) 
        #0x0C: int (4 bytes) 
        #0x0D: float (4 bytes) 
        #0x0E: double (8 bytes)
        dtype = f.read(1)
        assert(dtype == b'\x08')      
        #The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
        n_dims = to_int(f.read(1))
        # print(int.from_bytes(n_dims,"big"))
        #The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
        dims = [to_int(f.read(4)) for i in range(n_dims)]
        n_data = 1
        for d in dims:
            n_data *= d
        data = f.read(n_data)
        ## check that there was exactly the amount of data expected until EOF
        assert(len(data) == n_data)
        assert(f.read(1) == b'')
        data = np.array([to_int(data[i:(i + 1)]) for i in range(len(data))],dtype=np.uint32,order='C').reshape(dims)
        return data



def show_digit(indx, test_data=False):
    if test_data:
        plt.imshow(mnist[2][indx], cmap='Greys')
        print(mnist[3][indx])
    else:
        plt.imshow(mnist[0][indx], cmap='Greys')
        print(mnist[1][indx])

files_path = [
    r'Digits\train-images-idx3-ubyte\train-images.idx3-ubyte',
    r'Digits\train-labels-idx1-ubyte\train-labels.idx1-ubyte',
    r'Digits\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte',
    r'Digits\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte',
]

mnist = [read_data(path) for path in files_path]


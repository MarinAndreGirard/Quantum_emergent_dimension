import os
from scipy.sparse import coo_matrix, kron
import numpy as np
from itertools import product, combinations
import random
from scipy import sparse

paulis = [np.eye(2), np.array([[0,1],[1,0]]), 1j*np.array([[0,-1],[1,0]]), np.array([[1,0],[0,-1]])]
paulis_sparse = [coo_matrix(p, dtype='complex128') for p in paulis]

def operator_from_indexes(indexes, dtype='float64'):
    """
    indexes : list of pauli string indexes (eg [0,1,2,0,3])
    return : coo_matrix representing a pauli string (eg 1XY1Z)
    """
    op = paulis_sparse[indexes[0]]
    for i in indexes[1:]:
        op = kron(op, paulis_sparse[i], format='coo')
    if dtype=='float64':
        op = op.real
    return coo_matrix(op, dtype=dtype)



def buildH(folder, N, k):
    """
    build the hamitonian corresponding to the kth sample of size N from a particular folder
    folder is of the form ensemble_model
    """
    path = '{}/{}'.format(folder, N)
    labels = np.loadtxt(path+'/labels.txt')
    labels = np.array(labels, dtype=int)
    couplings = np.loadtxt(path+'/couplings.txt')[k] #select the kth sample from the couplings
    H = np.zeros((2**N,2**N))
    for i in range(len(labels)):
        tau = operator_from_indexes(labels[i])
        H[tau.row, tau.col] += couplings[i]*tau.data
    return H

if __name__ =='__main__':
    from scipy.linalg import eigvalsh
    import matplotlib.pyplot as plt
    N = 12
    k = 0
    folder = 'goe_xxyyzz'
    Hlocal = buildH(folder, N, k)
    e0 = eigvalsh(Hlocal)
    plt.hist(e0,bins=100,histtype='step', label='localized')
    plt.legend()
    plt.show()

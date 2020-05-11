import scipy.io as spio
import numpy as np
import csv

mat = spio.loadmat('data_LDMOS.mat')
in_ext = mat['in_extraction']
out_ext = mat['out_extraction']
in_val = mat['in_validation']
out_val = mat['out_validation']

M = 1
P = 5
def x_mp(entrada, M, P):
    modulo_entrada = np.absolute(entrada)
    X_MP = np.zeros((len(entrada),P*(M+1)),dtype=complex)
    for i in range(M+1,len(entrada)):
        for m in range(M+1):
            for p in range(1,P+1):
                j = ((m*P)-1)+p
                X_MP[i][j] = entrada[i-m][0]*((modulo_entrada[i-m][0])**(p-1))
    return X_MP

X_ext = x_mp(in_ext,M,P)
X_ext2 = X_ext[M+3:len(X_ext)-(M+3)][:]
out_ext2 = out_ext[M+3:len(out_ext)-(M+3)][:]
coefs = np.linalg.lstsq(X_ext2,out_ext2,rcond=-1)
coefs = coefs[0]

X_val = x_mp(in_val,M,P)
X_val2 = X_val[M+3:len(X_val)-(M+3)][:]
out_calc_mat_cmplx = X_val2@coefs
print(out_calc_mat_cmplx)
#np.savetxt('data.csv', X_ext, delimiter=',')

import numpy as np
import joblib

def calcKL_gauss(pos1, pos2,epsilon=0.001):
    N = pos1[0].shape[0]
    f2 = pos2[0]
    f1 = pos1[0]
    S2 = pos2[1] + epsilon * np.eye(N)
    S1 = pos1[1] + epsilon * np.eye(N)
    S2_inv = np.linalg.inv(S2)
    S = S2_inv @ S1
    trace = np.trace(S)
    logdet = np.log(np.linalg.det(S))
    se = np.trace((f2 - f1).T @ S2_inv @ (f2 - f1))
    KL = 0.5 * (trace - logdet + se - N)
    return KL

def calcKL_pq_fast(pos_old,new_output,beta):
    if pos_old[0].ndim != 3:
        m = pos_old[0][-1,-1]
        k = pos_old[1][-1,-1]
    else:
        m = pos_old[0][:,-1,-1]
        k = pos_old[1][:,-1,-1]
    trace = beta*k
    logdet = np.log(1+beta*k)
    se = beta*k/(k+1/beta)*(new_output-m)**2
    KL = 0.5*(trace - logdet + se)
    return KL


def calcKL_qp_fast(pos_old,new_output,beta):
    if pos_old[0].ndim != 3:
        m = pos_old[0][-1,-1]
        k = pos_old[1][-1,-1]
    else:
        m = pos_old[0][:,-1,-1]
        k = pos_old[1][:,-1,-1]
    trace = k/(k+1/beta)
    logdet = np.log(1+beta*k)
    se = k/((k+1/beta)**2)*(new_output-m)**2
    KL = 0.5*(-trace + logdet + se)
    return KL

def create_grid(node_size,dim,x_range):
    x = np.linspace(x_range[0,0], x_range[0,1], node_size)[:, None]
    y = np.linspace(x_range[1,0], x_range[1,1], node_size)[:, None]
    node_x, node_y = np.meshgrid(x, y)
    grid = np.zeros((node_size, node_size, dim))
    grid[:, :, 0] = node_x
    grid[:, :, 1] = node_y
    node = grid.reshape(node_size*node_size,dim)

    return [grid,node]

def get_topn_data(X,y,p=0.5):
    topn = int(p * y.shape[0])
    sorted_indecies = np.argsort(y, axis=0)
    topn_X = X[sorted_indecies[:topn, 0]]
    topn_y = y[sorted_indecies[:topn, 0]]
    return topn_X,topn_y

def serialize(obj,save_name):
    with open(save_name, mode='wb') as f:
        joblib.dump(obj, f)


def deserialize(save_name):
    with open(save_name, mode='rb') as f:
        obj = joblib.load(f)
    return obj

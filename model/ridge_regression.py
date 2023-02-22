import numpy as np
from sklearn.linear_model import Ridge


class RidgeRegression(Ridge):
    def __init__(self, *args, basis_size=10, length=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis_size = basis_size
        self.length = length
        self.basis_func = self.additive_RBF_function

    def fit(self,X,y,sample_weight=None):
        self.X = X
        self.x_range = [X.min(),X.max()]
        self.Phi = self.basis_func(self.X)
        super().fit(self.Phi,y,sample_weight)

    def predict(self, X):
        Phi = self.basis_func(X)
        return super().predict(Phi)

    def additive_poly_function(self,inputs):
        Phi = np.zeros((inputs.shape[0],inputs.shape[1],self.basis_size))
        for m in range(self.basis_size):
            Phi[:,:,m] = inputs**m
        Phi = Phi.reshape(self.X.shape[0],self.X.shape[1]*self.basis_size)
        return Phi

    def additive_RBF_function(self,inputs):
        Phi = np.zeros((inputs.shape[0],inputs.shape[1],self.basis_size))
        for d in range(inputs.shape[1]):
            [node, step] = np.linspace(self.x_range[0], self.x_range[1], self.basis_size, retstep=True)
            dist = ((inputs[:, d][:,None] - node[None, :]) ** 2)
            Phi[:,d,:] = np.exp(-1 / (2 * self.length ** 2) * dist)
        Phi = Phi.reshape(inputs.shape[0],inputs.shape[1]*self.basis_size)
        return Phi

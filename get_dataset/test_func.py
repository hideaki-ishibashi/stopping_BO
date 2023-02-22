import numpy as np
import itertools

def ackley(X):
    dim = X.shape[1]
    a = 20
    b = 0.2
    c = 2*np.pi
    y = -a*np.exp(-b*np.sqrt((X**2).sum(axis=1)/dim))-np.exp(np.cos(c*X).sum(axis=1)/dim)+a+np.e
    return y

def hartman3D(X):
    a = np.array([1.0,1.2,3.0,3.2])
    A = np.array([[3.0,10.0,30],
                  [0.1,10.0,35],
                  [3.0,10.0,30],
                  [0.1,10.0,35]])
    P = 1e-4*np.array([[3689.0,1170.0,2673.0],
                       [4699.0,4387.0,7470.0],
                       [1091.0,8732.0,5547.0],
                       [381.0,5743.0,8828.0]])
    y = -(a[None,:]*np.exp(-(A[None,:,:]*(X[:,None,:]-P[None,:,:])**2).sum(axis=2))).sum(axis=1)
    return y

def hartman6D(X):
    a = np.array([1.0,1.2,3.0,3.2])
    A = np.array([[10.0,3.0,17.0,3.5,1.7,8],
                  [0.05,10.0,17.0,0.1,8.0,14.0],
                  [3.0,3.5,1.7,10.0,17.0,8.0],
                  [17.0,8.0,0.05,10.0,0.1,14]])
    P = 1e-4*np.array([[1312.0,1696.0,5569.0,124.0,8283.0,5886.0],
                       [2329.0,4135.0,8307.0,3736.0,1004.0,9991.0],
                       [2348.0,1451.0,3522.0,2883.0,3047.0,6650.0],
                       [4047.0,8828.0,8732.0,5743.0,1091.0,381.0]])
    y = -(a[None,:]*np.exp(-(A[None,:,:]*(X[:,None,:]-P[None,:,:])**2).sum(axis=2))).sum(axis=1)
    return y

def styblinski_tang(X):
    y = 0.5*(X**4-16*X**2+5*X).sum(axis=1)
    return y

def alpine01(X):
    y = np.sum(np.abs(X*np.sin(X)+0.1*X),axis=1)
    return y

def bukin06(X):
    y = 100*np.sqrt(np.abs(X[:,1]-0.01*X[:,0]**2))+0.01*np.abs(X[:,0]+10)
    return y

def cross_in_tray(X):
    y = -0.0001*(np.abs(np.sin(X[:,0])*np.sin(X[:,1])*np.exp(np.abs(100-np.sqrt(X[:,0]**2+X[:,1]**2)/np.pi)))+1)**0.1
    return y

def drop_wave(X):
    y = -(1+np.cos(12*np.sqrt(X[:,0]**2+X[:,1]**2)))/(0.5*(X[:,0]**2+X[:,1]**2)+2)
    return y

def eggholder(X):
    y = -(X[:,1]+47)*np.sin(np.sqrt(np.abs(X[:,1]+X[:,0]/2+47)))-X[:,0]*np.sin(np.sqrt(np.abs(X[:,0]-(X[:,1]+47))))
    return y

def holder_table(X):
    y = -np.abs(np.sin(X[:,0])*np.cos(X[:,1])*np.exp(np.abs(1-np.sqrt((X**2).sum(axis=1))/np.pi)))
    return y

def sphere(X):
    y = (X**2).sum(axis=1)
    return y

def booth(X):
    y = (X[:,0]+2*X[:,1]-7)**2+(2*X[:,0]+X[:,1]-5)**2
    return y

def matyas(X):
    y = 0.26*((X**2).sum(axis=1))-0.48*X[:,0]*X[:,1]
    return y

def six_hump_camel(X):
    y = (4-2.1*X[:,0]**2+X[:,0]**4/3)*X[:,0]**2+X[:,0]*X[:,1]+(-4+4*X[:,1]**2)*X[:,1]**2
    return y

def easom(X):
    y = -np.cos(X[:, 0])*np.cos(X[:, 1])*np.exp(-(X[:, 0] - np.pi)**2-(X[:, 1] - np.pi)**2)
    return y

def rosenbrock(X):
    y = 0
    for d in range(X.shape[1]-1):
        y += 100*(X[:, d+1] - X[:, d]**2)**2+(X[:, d] - 1)**2
    return y


def get_config(function_name):
    if function_name is "ackley_4":
        dim = 4
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = -32.768
            x_range[d, 1] = 32.768
        x_min = np.zeros((1,2))
    elif function_name is "ackley_6":
        dim = 6
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = -32.768
            x_range[d, 1] = 32.768
        x_min = np.zeros((1, 2))
    elif function_name is "alpine01_4":
        dim = 4
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = -10
            x_range[d, 1] = 10
        x_min = np.zeros(dim)[None, :]
    elif function_name is "alpine01_6":
        dim = 6
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = -10
            x_range[d, 1] = 10
        x_min = np.zeros(dim)[None, :]
    elif function_name is "alpine01_10":
        dim = 6
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = -10
            x_range[d, 1] = 10
        x_min = np.zeros(dim)[None, :]
    elif function_name is "hartman3D":
        dim = 3
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = 0
            x_range[d, 1] = 1
        x_min = np.array([0.114614,0.555649,0.852547])[None,:]
    elif function_name is "hartman6D":
        dim = 6
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = 0
            x_range[d, 1] = 1
        x_min = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None, :]
    elif function_name is "styblinski_tang":
        dim = 3
        x_range = np.zeros((dim, 2))
        for d in range(dim):
            x_range[d, 0] = -5
            x_range[d, 1] = 5
        x_min = -2.93534*np.ones(dim)[None,:]
    elif function_name is "bukin":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0,0] = -15
        x_range[0,1] = -5
        x_range[1,0] = -3
        x_range[1,1] = 3
        x_min = np.array([-10,1])[None,:]
    elif function_name is "cross_in_tray":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0,0] = -10
        x_range[0,1] = 10
        x_range[1,0] = -10
        x_range[1,1] = 10
        x_min = np.array([[-1.3491,-1.3491],[1.3491,-1.3491],[-1.3491,1.3491],[1.3491,1.3491]])
    elif function_name is "drop_wave":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -5.12
        x_range[0, 1] = 5.12
        x_range[1, 0] = -5.12
        x_range[1, 1] = 5.12
        x_min = np.array([0,0])[None, :]
    elif function_name is "eggholder":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -512
        x_range[0, 1] = 512
        x_range[1, 0] = -512
        x_range[1, 1] = 512
        x_min = np.array([512, 404.2319])[None,:]
    elif function_name is "holder_table":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -10
        x_range[0, 1] = 10
        x_range[1, 0] = -10
        x_range[1, 1] = 10
        x_min = np.array([[-8.05502, -9.66459],[-8.05502, 9.66459],[8.05502, -9.66459],[8.05502, 9.66459]])
    elif function_name is "sphere":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -5.12
        x_range[0, 1] = 5.12
        x_range[1, 0] = -5.12
        x_range[1, 1] = 5.12
        x_min = np.array([0,0])[None, :]
    elif function_name is "booth":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -10
        x_range[0, 1] = 10
        x_range[1, 0] = -10
        x_range[1, 1] = 10
        x_min = np.array([1, 3])[None, :]
    elif function_name is "matyas":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -10
        x_range[0, 1] = 10
        x_range[1, 0] = -10
        x_range[1, 1] = 10
        x_min = np.array([0, 0])[None, :]
    elif function_name is "six_hump_camel":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -3
        x_range[0, 1] = 3
        x_range[1, 0] = -2
        x_range[1, 1] = 2
        x_min = np.array([[0.0898, -0.7126],[-0.0898, 0.7126]])
    elif function_name is "easom":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[0, 0] = -100
        x_range[0, 1] = 100
        x_range[1, 0] = -100
        x_range[1, 1] = 100
        x_min = np.array([[np.pi, np.pi]])
    elif function_name is "rosenbrock":
        dim = 2
        x_range = np.zeros((dim, 2))
        x_range[:, 0] = -5
        x_range[:, 1] = 10
        x_min = np.ones(dim)[None, :]
    else:
        return []
    return x_range, x_min


def get_samples(function_name=None):
    sample_size = 10000
    x_range, x_min = get_config(function_name)
    X = np.zeros((sample_size, x_range.shape[0]))
    for d in range(x_range.shape[0]):
        X[:, d] = np.random.uniform(x_range[d, 0], x_range[d, 1], sample_size)
    return X


def get_output(X, x_min, function_name, noise_var=0.1):
    f = function(X, x_min, function_name)
    y = f + np.random.normal(0, noise_var, X.shape[0])
    return y


def get_dataset(sample_size, function_name=None, noise_var=0.1):
    x_range, x_min = get_config(function_name)
    X = np.zeros((sample_size, x_range.shape[0]))
    for d in range(x_range.shape[0]):
        X[:, d] = np.random.uniform(x_range[d, 0], x_range[d, 1], sample_size)
    f = function(X, x_min, function_name)[:, None]
    noise = np.random.normal(0, np.sqrt(noise_var), sample_size)
    y = f + noise[:, None]
    return [X, y, x_min]


def function(X,x_min,function_name):
    sample = get_samples(function_name)
    if function_name is "ackley":
        return (ackley(X) - ackley(x_min)[0])
    elif function_name is "alpine01":
        return (alpine01(X) - alpine01(x_min)[0])
    elif function_name is "hartman3D":
        return (hartman3D(X) - hartman3D(x_min)[0])
    elif function_name is "hartman6D":
        return (hartman6D(X) - hartman6D(x_min)[0])
    elif function_name is "styblinski_tang":
        return styblinski_tang(X)
    elif function_name is "bukin":
        return (bukin06(X) - bukin06(x_min)[0])
    elif function_name is "cross_in_tray":
        return (cross_in_tray(X) - cross_in_tray(x_min)[0])
    elif function_name is "drop_wave":
        return (drop_wave(X) - drop_wave(x_min)[0])
    elif function_name is "eggholder":
        return (eggholder(X) - eggholder(x_min)[0])
    elif function_name is "holder_table":
        return (holder_table(X) - holder_table(x_min)[0])
    elif function_name is "sphere":
        return (sphere(X) - sphere(x_min)[0])
    elif function_name is "booth":
        return (booth(X) - booth(x_min)[0])
    elif function_name is "matyas":
        return (matyas(X) - matyas(x_min)[0])
    elif function_name is "six_hump_camel":
        return (six_hump_camel(X) - six_hump_camel(x_min)[0])
    elif function_name is "easom":
        return (easom(X) - easom(x_min)[0])
    elif function_name is "rosenbrock":
        return (rosenbrock(X) - rosenbrock(x_min)[0])
    else:
        return None

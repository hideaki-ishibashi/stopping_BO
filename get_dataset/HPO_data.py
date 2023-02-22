import numpy as np
import pandas as pd
import random
from sklearn.datasets import fetch_california_housing


def get_uci_dataset(data_name):
    # classification
    if data_name == "skin":
        dataset = np.loadtxt("UCI_data/skin/Skin_NonSkin.txt")
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1].astype("int")-1
        n_labels = len(np.unique(output))
        output = np.eye(n_labels)[output]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies,0].astype("int32")]

    if data_name == "electrical_grid_stability":
        df = pd.read_csv('UCI_data/electrical_grid_stability/electrical_grid_stability.csv', index_col=0)
        dataset = df.values
        input = dataset[:,:-2]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1].astype("int")
        n_labels = len(np.unique(output))
        output = np.eye(n_labels)[output]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies,0].astype("int32")]

    if data_name == "HTRU2":
        df = pd.read_csv('UCI_data/HTRU2/HTRU_2.csv', index_col=0)
        dataset = df.values
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1].astype("int")-1
        n_labels = len(np.unique(output))
        output = np.eye(n_labels)[output]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies,0].astype("int32")]

    # regression
    if data_name == "power_plant":
        dataset = np.loadtxt("UCI_data/power_plant/ccpp.txt")
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1]
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "gas_turbine":
        df = pd.read_csv('UCI_data/gas_turbine/gt_2011.csv', index_col=0)
        dataset1 = df.values
        df = pd.read_csv('UCI_data/gas_turbine/gt_2012.csv', index_col=0)
        dataset2 = df.values
        df = pd.read_csv('UCI_data/gas_turbine/gt_2013.csv', index_col=0)
        dataset3 = df.values
        df = pd.read_csv('UCI_data/gas_turbine/gt_2014.csv', index_col=0)
        dataset4 = df.values
        df = pd.read_csv('./UCI_data/gas_turbine/gt_2015.csv', index_col=0)
        dataset5 = df.values
        dataset = np.concatenate([dataset1,dataset2,dataset3,dataset4,dataset5],axis=0)
        input = np.concatenate([dataset[:,:7],dataset[:,7:]],axis=1)
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,7]
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "protein":
        dataset = np.loadtxt("UCI_data/protein/CASP.txt")
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1]
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "california_housing":
        housing = fetch_california_housing()
        input = housing["data"]
        output = housing["target"]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

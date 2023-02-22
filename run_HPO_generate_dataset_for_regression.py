from model.stopping_criteria import *
from utils import bo_utils
from get_dataset import HPO_data
from tqdm import tqdm
import random
import os
import itertools


def main():
    seed = 1
    model_names = ["ridge", "svr", "rfr"]
    data_names = ["gas_turbine", "power_plant", "protein"]
    for model_name in model_names:
        for data_name in data_names:
            np.random.seed(seed)
            random.seed(seed)
            dir = "result/HPO_experiments/" + data_name + "/" + model_name+"/"
            os.makedirs(dir, exist_ok=True)

            # get training and test data for predictive model
            train_size = 2000
            test_size = 2000
            dataset = HPO_data.get_uci_dataset(data_name)
            trainset = []
            trainset.append(dataset[0][:train_size, :])
            trainset.append(dataset[1][:train_size])
            whole_size = dataset[0].shape[0]
            print(whole_size)
            if test_size >= whole_size - train_size:
                test_size = whole_size - train_size
            print(test_size)
            testset = []
            testset.append(dataset[0][train_size:train_size + test_size, :])
            testset.append(dataset[1][train_size:train_size + test_size])
            # get model, space of hyper-parameter and evaluation metric of the predicive model
            model, bounds, param_dim, param_range, metric = bo_utils.set_model(model_name)
            params = np.array(list(itertools.product(*param_range)))
            rmses = np.zeros(params.shape[0])
            stds = np.zeros(params.shape[0])
            test_scores = np.zeros(params.shape[0])
            for n, param in enumerate(tqdm(params)):
                rmses[n], stds[n], test_scores[n] = bo_utils.model_evaluation(param,model,trainset,testset,metric=metric)
            np.savetxt(dir+"discretized_param.txt", params)
            np.savetxt(dir+"discretized_rmse.txt", rmses)
            np.savetxt(dir+"discretized_std.txt", stds)
            np.savetxt(dir+"discretized_test_score.txt", test_scores)

if __name__ == "__main__":
    main()

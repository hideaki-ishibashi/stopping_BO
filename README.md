# Code for A stopping criterion for Bayesian optimization by the gap of expected minimum simple regrets.

## Installation
Our code uses Python3.10.2 and the following packages:
- cycler          0.11.0  Composable style cycles
- cython          0.29.28 The Cython compiler for writing C extensions for the Python language.
- decorator       5.1.1   Decorators for Humans
- fonttools       4.32.0  Tools to manipulate font files
- gpy             1.10.0  The Gaussian Process Toolbox
- gpyopt          1.2.6   The Bayesian Optimization Toolbox
- joblib          1.1.0   Lightweight pipelining with Python functions
- kiwisolver      1.4.2   A fast implementation of the Cassowary constraint solver
- matplotlib      3.5.1   Python plotting package
- numpy           1.22.3  NumPy is the fundamental package for array computing with Python.
- packaging       21.3    Core utilities for Python packages
- pandas          1.4.2   Powerful data structures for data analysis, time series, and statistics
- paramz          0.9.5   The Parameterization Framework
- pillow          9.1.0   Python Imaging Library (Fork)
- pyparsing       3.0.8   pyparsing module - Classes and methods to define and execute parsing grammars
- python-dateutil 2.8.2   Extensions to the standard Python datetime module
- pytz            2022.1  World timezone definitions, modern and historical
- scikit-learn    1.0.2   A set of python modules for machine learning and data mining
- scipy           1.8.0   SciPy: Scientific Library for Python
- setuptools-scm  6.4.2   the blessed package to manage your versions by scm tags
- six             1.16.0  Python 2 and 3 compatibility utilities
- sklearn         0.0     A set of python modules for machine learning and data mining
- threadpoolctl   3.1.0   threadpoolctl
- tomli           2.0.1   A lil' TOML parser
- tqdm            4.64.0  Fast, Extensible Progress Meter

If you use poetry and pyenv, you can install the packages by running:

```
pyenv install 3.10.2
git clone https://github.com/hideaki-ishibashi/stopping_BO.git
cd stopping_BO
poetry env use ~/.pyenv/versions/3.10.2/bin/python
poetry install
```
Otherwise, you can install the packages by running:

```
git clone https://github.com/hideaki-ishibashi/stopping_BO.git
cd stopping_BO
pip install -r requirements.txt
```

## A brief overview of construction of our code

- `run_test_funct_optimization.py` and `run_HPO.py`
  - main code for BO search
- `plot_BO_test_function.py` and `plot_HPO.py`
  - code for visualization
- `run_HOP_generate_dataset_for_classification.py` and `run_HOP_generate_dataset_for_regression.py`
  - code for generating HPO dataset
- model folder
  - `GPyOptBO.py`
    - code modifying the GPyOpt package
  - `stopping_criterion.py`
    - code defining the stopping criterion
  - `ridge_regression.py`
    - code for ridge regression with additive RBF basis
  - `logistic_regression.py`
    - code for logistic regression with additive RBF basis
- utils folder
  - `utils.py`
    - utilities for those other than GPyOpt
  - `bo_utils.py`
    - settings for GPyOpt
  - `draw_result.py`
    - code for visualization
- UCI_data
    - UCI dataset
- get_dataset
    - function to retrieve test functions and datasets from UCI

## Usage
- main code
    - The experiment of test function can be reproduced by changing the function_name in run_BO_test_function.py. Note that budget is set to 400 in holder_table function and set to 200 in other functions.
        - function_name
            - "holder_table", "cross_in_tray", "six_hump_camel", "easom", "rosenbrock", "booth". 
    - The experiment of test function can be reproduced by changing the model_name and data_name in run_HPO.py.
        - model_name
            - classification : "logistic","svc","rfc". 
            - regression : "ridge","svr","rfr".
        - data_name
            - classification : "skin","HTRU2","electrical_grid_stability".
            - regression : "gas_turbine", "power_plant","protein".
- `stopping_criterion.py`
    - When definning the stopping criterion, threshold and budget are required.
    - check_threshold function calculates a value of each stopping criterion and determines if the threshold has fallen below a threshold.
- How to use GPyOpt : https://github.com/SheffieldML/GPyOpt

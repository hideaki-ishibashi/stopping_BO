import GPy
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, AcquisitionLCB_MCMC, AcquisitionLP, AcquisitionEntropySearch
from GPyOpt.core.bo import BO
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.experiment_design import initial_design
from GPyOpt.core.task.space import Design_space, bounds_to_space
from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.core.task.cost import CostModel
from GPyOpt.util.arguments_manager import ArgumentsManager
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC
from GPyOpt.models.rfmodel import RFModel
from GPyOpt.models.warpedgpmodel import WarpedGPModel
from GPyOpt.models.input_warped_gpmodel import InputWarpedGPModel
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.optimization.acquisition_optimizer import ContextManager
try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass
import GPyOpt


import warnings
warnings.filterwarnings("ignore")

class GPyOptBO(BO):
    """
    Main class to initialize a Bayesian Optimization method.
    :param f: function to optimize. It should take 2-dimensional numpy arrays as input and return 2-dimensional outputs (one evaluation per row).
    :param domain: list of dictionaries containing the description of the inputs variables (See GPyOpt.core.task.space.Design_space class for details).
    :param constraints: list of dictionaries containing the description of the problem constraints (See GPyOpt.core.task.space.Design_space class for details).
    :cost_withGradients: cost function of the objective. The input can be:
        - a function that returns the cost and the derivatives and any set of points in the domain.
        - 'evaluation_time': a Gaussian process (mean) is used to handle the evaluation cost.
    :model_type: type of model to use as surrogate:
        - 'GP', standard Gaussian process.
        - 'GP_MCMC', Gaussian process with prior in the hyper-parameters.
        - 'sparseGP', sparse Gaussian process.
        - 'warperdGP', warped Gaussian process.
        - 'InputWarpedGP', input warped Gaussian process
        - 'RF', random forest (scikit-learn).
    :param X: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y: 2d numpy array containing the initial outputs (one per row) of the model.
    :initial_design_numdata: number of initial points that are collected jointly before start running the optimization.
    :initial_design_type: type of initial design:
        - 'random', to collect points in random locations.
        - 'latin', to collect points in a Latin hypercube (discrete variables are sampled randomly.)
    :acquisition_type: type of acquisition function to use.
        - 'EI', expected improvement.
        - 'EI_MCMC', integrated expected improvement (requires GP_MCMC model).
        - 'MPI', maximum probability of improvement.
        - 'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model).
        - 'LCB', GP-Lower confidence bound.
        - 'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :exact_feval: whether the outputs are exact (default False).
    :acquisition_optimizer_type: type of acquisition function to use.
        - 'lbfgs', L-BFGS.
        - 'DIRECT', Dividing Rectangles.
        - 'CMA', covariance matrix adaptation.
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param evaluator_type: determines the way the objective is evaluated (all methods are equivalent if the batch size is one)
        - 'sequential', sequential evaluations.
        - 'random', synchronous batch that selects the first element as in a sequential policy and the rest randomly.
        - 'local_penalization', batch method proposed in (Gonzalez et al. 2016).
        - 'thompson_sampling', batch method using Thompson sampling.
    :param batch_size: size of the batch in which the objective is evaluated (default, 1).
    :param num_cores: number of cores used to evaluate the objective (default, 1).
    :param verbosity: prints the models and other options during the optimization (default, False).
    :param maximize: when True -f maximization of f is done by minimizing -f (default, False).
    :param **kwargs: extra parameters. Can be used to tune the current optimization setup or to use deprecated options in this package release.
    .. Note::   The parameters bounds, kernel, numdata_initial_design, type_initial_design, model_optimize_interval, acquisition, acquisition_par
                model_optimize_restarts, sparseGP, num_inducing and normalize can still be used but will be deprecated in the next version.
    """

    def __init__(self, f, domain = None, constraints = None, cost_withGradients = None, model_type = 'GP', X = None, Y = None,
    	initial_design_numdata = 5, initial_design_type='random', acquisition_type ='EI', normalize_Y = True,
        exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1, evaluator_type = 'sequential',
        batch_size = 1, num_cores = 1, verbosity=False, verbosity_model = False, maximize=False, de_duplication=False, **kwargs):
        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs

        # --- Handle the arguments passed via kwargs
        self.problem_config = ModifiedArgumentsManager(kwargs)

        # --- CHOOSE design space
        self.constraints = constraints
        self.domain = domain
        self.space = Design_space(self.domain, self.constraints)

        # --- CHOOSE objective function
        self.maximize = maximize
        if 'objective_name' in kwargs:
            self.objective_name = kwargs['objective_name']
        else:
            self.objective_name = 'no_name'
        self.batch_size = batch_size
        self.num_cores = num_cores
        if f is not None:
            self.f = self._sign(f)
            self.objective = SingleObjective(self.f, self.batch_size,self.objective_name)
        else:
            self.f = None
            self.objective = None

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type  = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type. If an instance of a GPyOpt model is passed (possibly user defined), it is used.
        self.model_type = model_type
        self.exact_feval = exact_feval  # note that this 2 options are not used with the predefined model
        self.normalize_Y = normalize_Y

        if 'model' in self.kwargs:
            if isinstance(kwargs['model'], GPyOpt.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User defined model used.'
                print('Using a model defined by the used.')
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type

        # This states how the discrete variables are handled (exact search or rounding)
        kwargs.update({ 'model' : self.model })
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(self.space, self.acquisition_optimizer_type, **kwargs)  ## more arguments may come here

        # --- CHOOSE acquisition function. If an instance of an acquisition is passed (possibly user defined), it is used.
        self.acquisition_type = acquisition_type

        if 'acquisition' in self.kwargs:
            if isinstance(kwargs['acquisition'], GPyOpt.acquisitions.AcquisitionBase):
                self.acquisition = kwargs['acquisition']
                self.acquisition_type = 'User defined acquisition used.'
                print('Using an acquisition defined by the used.')
            else:
                self.acquisition = self._acquisition_chooser()
        else:
            self.acquisition = self.acquisition = self._acquisition_chooser()


        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        super(GPyOptBO,self).__init__(  model                  = self.model,
                                                    space                  = self.space,
                                                    objective              = self.objective,
                                                    acquisition            = self.acquisition,
                                                    evaluator              = self.evaluator,
                                                    X_init                 = self.X,
                                                    Y_init                 = self.Y,
                                                    cost                   = self.cost,
                                                    normalize_Y            = self.normalize_Y,
                                                    model_update_interval  = self.model_update_interval,
                                                    de_duplication         = self.de_duplication)

    def _model_chooser(self):
        return self.problem_config.model_creator(self.model_type, self.exact_feval,self.space)

    def _acquisition_chooser(self):
        return self.problem_config.acquisition_creator(self.acquisition_type, self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients)

    def _evaluator_chooser(self):
        return self.problem_config.evaluator_creator(self.evaluator_type, self.acquisition, self.batch_size, self.model_type, self.model, self.space, self.acquisition_optimizer)

    def _init_design_chooser(self):
        """
        Initializes the choice of X and Y based on the selected initial design and number of points selected.
        """

        # If objective function was not provided, we require some initial sample data
        if self.f is None and (self.X is None or self.Y is None):
            raise InvalidConfigError("Initial data for both X and Y is required when objective function is not provided")

        # Case 1:
        if self.X is None:
            self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)
        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)

    def _sign(self,f):
         if self.maximize:
             f_copy = f
             def f(x):return -f_copy(x)
         return f

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """

        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        ### --- Activate de_duplication
        if self.de_duplication:
            zipped_X = self.X
            duplicate_manager = DuplicateManager(space=self.space, zipped_X=zipped_X, pending_zipped_X=pending_zipped_X, ignored_zipped_X=ignored_zipped_X)
        else:
            duplicate_manager = None

        ### We zip the value in case there are categorical variables
        return self.space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=duplicate_manager, context_manager= self.acquisition.optimizer.context_manager))

class ModifiedArgumentsManager(ArgumentsManager):
    def model_creator(self, model_type, exact_feval, space):
        """
        Model chooser from the available options. Extra parameters can be passed via **kwargs.
        """
        model_type = model_type
        exact_feval = exact_feval
        space = space

        kernel = self.kwargs.get('kernel',None)
        ARD = self.kwargs.get('ARD',False)
        verbosity_model = self.kwargs.get('verbosity_model',False)
        noise_var = self.kwargs.get('noise_var',None)
        fixed_noise_var = self.kwargs.get('fixed_noise_var',None)
        noise_range = self.kwargs.get('noise_range',None)
        model_optimizer_type = self.kwargs.get('model_optimizer_type','lbfgs')
        max_iters = self.kwargs.get('max_iters',1000)
        optimize_restarts = self.kwargs.get('optimize_restarts',5)

        # --------
        # --- Initialize GP model with MLE on the parameters
        # --------
        if model_type == 'GP' or model_type == 'sparseGP':
            if model_type == 'GP':
                sparse = False
            if model_type == 'sparseGP':
                sparse = True
            optimize_restarts = self.kwargs.get('optimize_restarts',5)
            num_inducing = self.kwargs.get('num_inducing',10)
            return GPModel(kernel, noise_var, exact_feval, model_optimizer_type, max_iters, optimize_restarts, sparse, num_inducing, verbosity_model, ARD)

        if model_type == 'modifiedGP':
            sparse = False
            optimize_restarts = self.kwargs.get('optimize_restarts',5)
            num_inducing = self.kwargs.get('num_inducing',10)
            return ModifiedGPModel(kernel, noise_var, exact_feval, model_optimizer_type, max_iters, optimize_restarts, sparse, num_inducing, verbosity_model, ARD, fixed_noise_var=fixed_noise_var, noise_range=noise_range)

        # --------
        # --- Initialize GP model with MCMC on the parameters
        # --------
        elif model_type == 'GP_MCMC':
            n_samples = self.kwargs.get('n_samples',10)
            n_burnin = self.kwargs.get('n_burnin',100)
            subsample_interval = self.kwargs.get('subsample_interval',10)
            step_size = self.kwargs.get('step_size',1e-1)
            leapfrog_steps = self.kwargs.get('leapfrog_steps',20)
            return GPModel_MCMC(kernel, noise_var, exact_feval, n_samples, n_burnin, subsample_interval, step_size, leapfrog_steps, verbosity_model)

        # --------
        # --- Initialize RF: values taken from default in scikit-learn
        # --------
        elif model_type =='RF':
            return RFModel(verbose=verbosity_model)

        # --------
        # --- Initialize WapedGP in the outputs
        # --------
        elif model_type =='warpedGP':
            return WarpedGPModel()

        # --------
        # --- Initialize WapedGP in the inputs
        # --------
        elif model_type == 'input_warped_GP':
            if 'input_warping_function_type' in self.kwargs:
                if self.kwargs['input_warping_function_type'] != "kumar_warping":
                    print("Only support kumar_warping for input!")

            # Only support Kumar warping now, setting it to None will use default Kumar warping
            input_warping_function = None
            optimize_restarts = self.kwargs.get('optimize_restarts',5)
            return InputWarpedGPModel(space, input_warping_function, kernel, noise_var,
                                      exact_feval, model_optimizer_type, max_iters,
                                      optimize_restarts, verbosity_model, ARD)


class ModifiedGPModel(GPModel):
    def __init__(self, *args, fixed_noise_var=1e-6, noise_range=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_noise_var = fixed_noise_var
        if noise_range is None:
            self.noise_range = [1e-9, 1e-6]
        else:
            self.noise_range = noise_range

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        if not self.sparse:
            self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var, mean_function=self.mean_function)
        else:
            self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing, mean_function=self.mean_function)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(self.fixed_noise_var, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(self.noise_range[0], self.noise_range[1], warning=False) #constrain_positive(warning=False)

    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = ModifiedGPModel(kernel = self.model.kern.copy(),
                            noise_var=self.noise_var,
                            exact_feval=self.exact_feval,
                            optimizer=self.optimizer,
                            max_iters=self.max_iters,
                            optimize_restarts=self.optimize_restarts,
                            verbose=self.verbose,
                            ARD=self.ARD,
                            noise_range=self.noise_range)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

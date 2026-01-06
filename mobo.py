""" MOBO loop functions """


# librairies


from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from analysis import true_kin, meas_kin, recon_kin
from simu import launch
from utils import *
from checkpoint import *


# geant4 wrapper


def wrapper(x):
    """
    Geant4 wrapper.

    Args:
        x (d-D torch): Geometrical parameters to optimize.
    
    Returns: 
        obj (float): objective function evaluation.
    """

    if isinstance(x, torch.Tensor):
        x = x.squeeze().tolist()

    merged_file = launch(x)

    # inputs: simple gun

    p_proton_in_true = np.sqrt(2.4988**2+99.964**2)
    pt_proton_in_true = 2.4988

    # inputs: sidis

    # p_proton_in_true = true_kin()['p_proton_in']
    # pt_proton_in_true = true_kin()['pt_proton_in']
    # p_lambda_out_true = true_kin()['p_lambda_out']
    # pt_lambda_out_true = true_kin()['pt_lambda_out']
    # t_lambda_true = true_kin()['t_lambda']

    # outputs: recon

    # p_recon_mean = recon_kin(merged_file)['p']
    # pt_recon_mean = recon_kin(merged_file)['pT']

    # objective /!\ botorch will maximize it (add minus if minimization is desired)

    if OBS_OBJ=='p':
        p_meas_mean = meas_kin(merged_file)['p']
        p_meas_err = meas_kin(merged_file)['p_err']
        obj = p_meas_mean #-abs(p_proton_in_true-p_meas_mean)/p_proton_in_true
        obj_err = p_meas_err
        print(f"[DEBUG OBJECTIVE FUNCTION] p = {p_meas_mean} +/- {p_meas_err}")

    elif OBS_OBJ=='pT':
        pt_meas_mean = meas_kin(merged_file)['pT']
        pt_meas_err = meas_kin(merged_file)['pT_err']
        obj = pt_meas_mean #-abs(pt_proton_in_true-pt_meas_mean)/pt_proton_in_true
        obj_err = pt_meas_err

    else:
        raise ValueError(
            f"[DEBUG mobo.wrapper] OBS_OBJ='{OBS_OBJ}' not correct. Choose 'p' or 'pT'."
        )

    return obj, obj_err


# objective function


def of(x_norm, bounds):
    """
    Objective function evaluation.
    
    Args:
        b (int): Bayesian iteration.
        x_norm (float): Normalized value of the geometrical parameter.
        bounds (Tensor 2D): Real bounds for the geometrical parameter.
    
    Return:
        (Tensor 1D): Objective function value.
    """
    x = denormalize(x_norm, bounds)
    y, y_err = wrapper(x)
    return torch.tensor([[y]], dtype=torch.double), y_err


# optimization loop function


def bayesian_optimization(bounds, n_init, n_iter):
    """
    Objective function evaluation.
    
    Args:
        bounds (Tensor 2D): Real bounds for the geometrical parameter.
        n_init (int): Real bounds for the geometrical parameter.
        n_iter (int): Real bounds for the geometrical parameter.
    
    Return:
        (Tensor 1D): Objective function value.
    """

    # dim

    d = bounds.shape[1]
    bounds_norm = torch.stack([torch.zeros(d), torch.ones(d)])

    # checkpoint loading

    state = load_latest_ckpt()

    # init

    y_err, converg_fct, converg_fct_err = [], [], []
    gp, mll = None, None

    # if no checkpoint to load

    if state is None:

        print(f"[DEBUG mobo.bayesian_optimization] Checkpoint status: start {n_init} random evals.")
        it_start = 0
        x_train_norm = torch.empty((0, d), dtype=torch.double)
        y_train = torch.empty((0, 1), dtype=torch.double)

        for i in range(n_init):

            # of

            x_new = torch.rand(1, d, dtype=torch.double)
            y_new, y_new_err = of(x_new, bounds)
            x_train_norm = torch.cat([x_train_norm, x_new], dim=0)
            y_train = torch.cat([y_train, y_new], dim=0)
            y_err.append(y_new_err)

            # convergence

            best_idx = torch.argmax(y_train).item()
            converg_fct.append(y_train[best_idx].item())
            converg_fct_err.append(y_err[best_idx])

            # checkpoint saving

            save_ckpt(i,
                      x_train_norm,
                      y_train,
                      y_err=y_err,
                      model=gp,
                      mll=mll,
                      bounds=bounds,
                      converg_fct=converg_fct,
                      converg_fct_err=converg_fct_err,
                      reason="init")

    # if last checkpoint to load

    else:

        it_start = int(state["it"]) + 1
        x_train_norm = state["x"].to(torch.double)
        y_train = state["y"].to(torch.double)
        y_err = state.get("y_err", []) 
        converg_fct = state.get("converg_fct", [y_train.max().item()])
        converg_fct_err = state.get("converg_fct_err", [0.0])
        print(f"[DEBUG mobo.bayesian_optimization] Checkpoint: restart from iteration {it_start}.")

    for it in range(it_start, n_iter):

        # checkpoint stop

        if should_stop():

            save_ckpt(it-1,
                      x_train_norm,
                      y_train,
                      y_err=y_err,
                      model=gp,
                      mll=mll,
                      bounds=bounds,
                      converg_fct=converg_fct,
                      converg_fct_err=converg_fct_err,
                      reason="time_budget")

            print("[DEBUG mobo.bayesian_optimization] Checkpoint: 24h clean exit.",flush=True)
            return

        # surrogate function

        gp = SingleTaskGP(x_train_norm,
                          y_train,
                          outcome_transform=Standardize(m=1)).to(torch.double)

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        fit_gpytorch_mll(mll)

        # acquisition function: qNEI

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        acq = qLogNoisyExpectedImprovement(model=gp,
                                           X_baseline=x_train_norm,
                                           sampler=sampler)

        candidate, _ = optimize_acqf(acq_function=acq,
                                     bounds=bounds_norm,
                                     q=1,
                                     num_restarts=20,
                                     raw_samples=100)

        # evaluation

        y_new, y_new_err = of(candidate, bounds)

        # update

        x_train_norm = torch.cat([x_train_norm, candidate], dim=0).to(torch.double)
        y_train = torch.cat([y_train, y_new], dim=0).to(torch.double)
        y_err.append(y_new_err)

        # plot

        _ = plot_mobo(it, gp, bounds, x_train_norm, y_train)

        # convergence

        best_idx = torch.argmax(y_train).item()
        converg_fct.append(y_train[best_idx].item())
        converg_fct_err.append(y_err[best_idx])
        _ = plot_convergence(converg_fct, converg_fct_err)

        # checkpoint

        save_ckpt(it,
                  x_train_norm,
                  y_train,
                  y_err=y_err,
                  model=gp,
                  mll=mll,
                  bounds=bounds,
                  converg_fct=converg_fct,
                  converg_fct_err=converg_fct_err,
                  reason="periodic")

    # final result

    best_idx = torch.argmax(y_train)
    best_x_norm = x_train_norm[best_idx]
    best_x = denormalize(best_x_norm, bounds)
    best_y = y_train[best_idx]
    print("\n Best result :")
    print(f"x = {best_x.numpy()}, y = {best_y.item():.4f}")
    return best_x, best_y


if __name__ == "__main__":

    for dirname in ["bayesian_iterations", "checkpoints", "fig", "logs", "simu"]:
        os.makedirs(dirname, exist_ok=True)

    BOUNDS = get_bounds_from_env()
    bayesian_optimization(BOUNDS, 1, N_ITER)

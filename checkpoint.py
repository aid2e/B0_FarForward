""" Checkpoint manager """

# librairies

import os, time, signal, sys, random, numpy as np, torch
from pathlib import Path
from utils import PROJECT_DIR

# global

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", f"{PROJECT_DIR}/checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
START_TIME = time.time()
_TIME_BUDGET = int(os.environ.get("TIME_BUDGET_SEC"))
DEADLINE = (START_TIME + _TIME_BUDGET) if _TIME_BUDGET > 0 else None

# checkpoint functions

def _ckpt_path(it):
    """ 
    Extract checkpoint path from iteration number.

    Args:
        it (int): Bayesian iteration number.

    Returns:
        (Path): Checkpoint path.
    """

    return CHECKPOINT_DIR / f"ckpt_{it}.pt"


def _to_double_cpu(t):
    """ 
    Extract checkpoint path from iteration number.

    Args:
        it (int): Bayesian iteration number.

    Returns:
        (Path): Checkpoint path.
    """

    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu().to(torch.double)
    return t


def save_ckpt(it, x, y, y_err,
              model=None, mll=None, 
              bounds=None,
              converg_fct=None, converg_fct_err=None, 
              extra=None, reason="periodic", keep=0):
    """ 
    Saving iteration.

    Args:
        it (int): Bayesian iteration number.
        x (torch): Bayesian trial.
        y (torch): Simulation result for the Bayesian trial.
        model (torch): Surrogate function.
        mll ():
        bounds (torch): Boundaries for the parameters to optimize.
        converg_fct (numpy array): Convergence fonction list.
        extra (): 
        reason (str): Reason for saving (for log out).
        keep (int): ...
    """

    # path

    path = _ckpt_path(it)
    tmp = path.with_suffix(".pt.tmp")

    # state definition

    state = {
        "it": int(it),
        "x": _to_double_cpu(x),
        "y": _to_double_cpu(y),
        "y_err": list(y_err) if y_err is not None else None,
        "bounds": _to_double_cpu(bounds) if hasattr(bounds, "shape") or torch.is_tensor(bounds) else bounds,
        "converg_fct": list(converg_fct) if converg_fct is not None else None,
        "converg_fct_err": list(converg_fct_err) if converg_fct_err is not None else None,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_py": random.getstate(),
        "reason": reason,
        "time": time.time(),
        "meta": {
            "hostname": getattr(os, "uname", lambda: type("U", (), {"nodename": ""}))().nodename if hasattr(os, "uname") else "",
            "pid": os.getpid(),
        },
    }

    if model is not None:
        state["model_state"] = model.state_dict() if hasattr(model, "state_dict") else model
    if mll is not None and hasattr(mll, "state_dict"):
        state["mll_state"] = mll.state_dict()
    if extra is not None:
        state["extra"] = extra

    # saving

    torch.save(state, tmp)
    os.replace(tmp, path)

    tmp_latest = LATEST_PATH.with_suffix(".pt.tmp")
    torch.save(state, tmp_latest)
    os.replace(tmp_latest, LATEST_PATH)

    if keep and keep > 0:
        files = sorted(CHECKPOINT_DIR.glob("ckpt_*.pt"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except Exception:
                pass

    print(f"[DEBUG checkpoint.save_ckpt] Achieved: saved iteration {it}. Reason: {reason}.", flush=True)


def load_latest_ckpt(reseed_rng=True):
    """ 
    Loading last iteration.

    Args:
        reseed_rng (bool): Change the seed or not.

    Returns:
        state (dict): State of loaded last iteration.
    """

    # path

    candidates = []
    if LATEST_PATH.exists():
        candidates.append(LATEST_PATH)
    candidates += sorted(CHECKPOINT_DIR.glob("ckpt_*.pt"), reverse=True)

    for p in candidates:
        
        try:
            
            state = torch.load(p, map_location="cpu")

            if state.get("x") is not None:
                state["x"] = state["x"].to(torch.double)
            if state.get("y") is not None:
                state["y"] = state["y"].to(torch.double)
            if state.get("bounds") is not None and torch.is_tensor(state["bounds"]):
                state["bounds"] = state["bounds"].to(torch.double)
            if reseed_rng:
                torch.set_rng_state(state["rng_torch"])
                np.random.set_state(state["rng_numpy"])
                random.setstate(state["rng_py"])
            print(f"[DEBUG checkpoint.load_latest_ckpt] Achieved: loaded {p.name} (iteration {state['it']})", flush=True)
            return state

        except Exception as e:
            
            print(f"[DEBUG checkpoint.load_latest_ckpt] Error: unable to load {p.name}: {e}", flush=True)
            continue
    
    return None


def should_stop(margin_sec=1800):
    """ 
    Stopper before 24h.

    Args:
        margin_sec (int): Margin to stop before 24h (seconds).

    Returns:
        (bool): Stop or not boolean variable.
    """
    if DEADLINE is None:
        return False
    return time.time() > (DEADLINE - max(0, int(margin_sec)))


def _sig_handler(signum, frame):
    try:
        save_ckpt(
            globals().get("it", -1),
            globals().get("x_train_norm"),
            globals().get("y_train"),
            globals().get("y_err"),
            globals().get("model") or globals().get("gp"),
            globals().get("mll"),
            globals().get("bounds"),
            converg_fct=globals().get("converg_fct"),
            converg_fct_err=globals().get("converg_fct_err"),
            reason=f"signal {signum}"
        )
    finally:
        print(f"[INFO] Caught signal {signum}. Checkpoint written. Exiting.", flush=True)
        sys.exit(0)

signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT,  _sig_handler)

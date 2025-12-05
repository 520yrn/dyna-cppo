# -*- coding: utf-8 -*-
"""
main.py

Training loops for:
- CPPO (paper reproduction)
- Dyna-CPPO (CPPO + short-horizon model-based rollouts)

Core steps (both variants):
- Collect on-policy rollout under current policy (Safety-Gymnasium):
    obs, reward, cost, terminated, truncated, info
- Compute dual-branch GAE: reward advantages A_r, cost advantages A_c,
  and bootstrapped returns for V_r, V_c
- E-step: solve for non-parametric optimal ratios v (q / p_pi) under
  cost & reverse-KL constraints
- M-step: update parametric policy π_θ to track v with a lower-bound
  clipped ratio (forward-KL control)
- Critic update: regress V_r to ret_r and V_c to ret_c

Dyna-CPPO adds:
- DynamicsModel: supervised training from real env transitions
- Short-horizon model rollouts from real states
- Combine real + model data for E-step / M-step / critic updates

Author: Ruining (CPPO origin) + Dyna-CPPO integration
"""

from __future__ import annotations
import os
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import safety_gymnasium  # environment backend

# ---- your local modules (adjust import paths if your tree is different) ----
from agents.policy import PolicyNet            # must expose: forward(obs)->(mu, std)
from agents.value  import ValueNet, CostNet    # simple MLP regressors for V_r and V_c
from agents.dynamics import DynamicsModel      # your lightweight dynamics model
from algo.advantages import compute_gae_dual   # dual-branch GAE (reward & cost)
from algo.estep      import estep_optimal_v    # solve target ratios v on a batch
from algo.mstep      import mstep_update       # track v with lower-bound clipped ratio
from utils.seed      import set_seed           # reproducibility
from utils.log       import Logger             # simple console/file logger


# -------------------------------
# helper: evaluate critics on obs
# -------------------------------
@torch.no_grad()
def _critic_forward(
    vr: nn.Module,
    vc: nn.Module,
    obs: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run V_r and V_c over a numpy batch of observations and return numpy arrays (on CPU)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    vr_pred = vr(obs_t).squeeze(-1)   # [N]
    vc_pred = vc(obs_t).squeeze(-1)   # [N]
    return vr_pred.cpu().numpy(), vc_pred.cpu().numpy()


# ---------------------------------------
# rollout: collect one on-policy trajectory (real env)
# ---------------------------------------
def collect_rollout(
    env,
    policy: nn.Module,
    vr: nn.Module,
    vc: nn.Module,
    horizon: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Collect one on-policy rollout of length T = horizon using the current policy π.
    Assumes Safety-Gymnasium step signature: (obs, reward, cost, terminated, truncated, info).

    Returns a dict of numpy arrays:
      'obs'        : [T, obs_dim] observations
      'next_obs'   : [T, obs_dim] next observations (for dynamics training)
      'act'        : [T, act_dim] actions sampled from π
      'old_logp'   : [T]          log π_old(a|s) at sampling time
      'rew'        : [T]          rewards
      'cost'       : [T]          costs
      'term'       : [T]          terminated flags
      'trunc'      : [T]          truncated flags
      'val_r'      : [T+1]        V_r(s_t) with bootstrap V_r(s_T)
      'val_c'      : [T+1]        V_c(s_t) with bootstrap V_c(s_T)
    """
    obs, _ = env.reset()

    obs_buf   = []
    next_buf  = []
    act_buf   = []
    logp_buf  = []
    rew_buf   = []
    cost_buf  = []
    term_buf  = []
    trunc_buf = []

    v_r_list = []
    v_c_list = []

    for t in range(horizon):
        obs_buf.append(obs.copy())

        # critics at s_t
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            vr_t = vr(obs_t).squeeze(0).item()
            vc_t = vc(obs_t).squeeze(0).item()
        v_r_list.append(vr_t)
        v_c_list.append(vc_t)

        # policy sample
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, std = policy(obs_t)
            dist = torch.distributions.Normal(mu, std)
            act_t = dist.sample()
            logp_t = dist.log_prob(act_t).sum(-1)
            act_np = act_t.squeeze(0).cpu().numpy()
            logp_np = logp_t.item()

        next_obs, rew, cost, terminated, truncated, info = env.step(act_np)

        act_buf.append(act_np)
        logp_buf.append(logp_np)
        rew_buf.append(float(rew))
        cost_buf.append(float(cost))
        term_buf.append(bool(terminated))
        trunc_buf.append(bool(truncated))
        next_buf.append(next_obs.copy())

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # bootstrap at s_T
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        vr_T = vr(obs_t).squeeze(0).item()
        vc_T = vc(obs_t).squeeze(0).item()

    obs_arr    = np.asarray(obs_buf,   dtype=np.float32)
    next_arr   = np.asarray(next_buf,  dtype=np.float32)
    act_arr    = np.asarray(act_buf,   dtype=np.float32)
    logp_arr   = np.asarray(logp_buf,  dtype=np.float32)
    rew_arr    = np.asarray(rew_buf,   dtype=np.float32)
    cost_arr   = np.asarray(cost_buf,  dtype=np.float32)
    term_arr   = np.asarray(term_buf,  dtype=np.bool_)
    trunc_arr  = np.asarray(trunc_buf, dtype=np.bool_)
    val_r_arr  = np.asarray(v_r_list + [vr_T], dtype=np.float32)
    val_c_arr  = np.asarray(v_c_list + [vc_T], dtype=np.float32)

    return dict(
        obs=obs_arr, next_obs=next_arr,
        act=act_arr, old_logp=logp_arr,
        rew=rew_arr, cost=cost_arr,
        term=term_arr, trunc=trunc_arr,
        val_r=val_r_arr, val_c=val_c_arr,
    )


# ---------------------------------------
# model-based rollout: DynamicsModel + policy
# ---------------------------------------
@torch.no_grad()
def collect_model_rollout(
    dyn_model: DynamicsModel,
    env_obs_init: np.ndarray,
    policy: nn.Module,
    vr: nn.Module,
    vc: nn.Module,
    horizon: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Use the learned dynamics model to generate a short model-based rollout,
    starting from a real environment observation env_obs_init.

    Returns a dict with the same structure as collect_rollout, so that
    they can be concatenated.
    """
    obs = env_obs_init.copy()

    obs_buf   = []
    next_buf  = []
    act_buf   = []
    logp_buf  = []
    rew_buf   = []
    cost_buf  = []
    term_buf  = []
    trunc_buf = []

    v_r_list = []
    v_c_list = []

    for t in range(horizon):
        obs_buf.append(obs.copy())

        # critics at s_t
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        v_r = vr(obs_t).squeeze(0).item()
        v_c = vc(obs_t).squeeze(0).item()
        v_r_list.append(v_r)
        v_c_list.append(v_c)

        # policy at s_t
        mu, std = policy(obs_t)
        dist = torch.distributions.Normal(mu, std)
        act_t = dist.sample()
        logp_t = dist.log_prob(act_t).sum(-1)
        act_np = act_t.squeeze(0).cpu().numpy()
        logp_np = logp_t.item()

        # dynamics one-step prediction
        pred = dyn_model(obs_t, act_t)
        next_obs_t = pred["next_obs"]
        next_obs_np = next_obs_t.squeeze(0).cpu().numpy()

        # optional reward / cost from model
        if dyn_model.predict_reward and "reward" in pred:
            rew = float(pred["reward"].item())
        else:
            rew = 0.0
        if dyn_model.predict_cost and "cost" in pred:
            cost = float(pred["cost"].item())
        else:
            cost = 0.0

        act_buf.append(act_np)
        logp_buf.append(logp_np)
        rew_buf.append(rew)
        cost_buf.append(cost)
        term_buf.append(False)
        trunc_buf.append(False)
        next_buf.append(next_obs_np.copy())

        obs = next_obs_np

    # bootstrap value at last state
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    vr_T = vr(obs_t).squeeze(0).item()
    vc_T = vc(obs_t).squeeze(0).item()

    obs_arr    = np.asarray(obs_buf,   dtype=np.float32)
    next_arr   = np.asarray(next_buf,  dtype=np.float32)
    act_arr    = np.asarray(act_buf,   dtype=np.float32)
    logp_arr   = np.asarray(logp_buf,  dtype=np.float32)
    rew_arr    = np.asarray(rew_buf,   dtype=np.float32)
    cost_arr   = np.asarray(cost_buf,  dtype=np.float32)
    term_arr   = np.asarray(term_buf,  dtype=np.bool_)
    trunc_arr  = np.asarray(trunc_buf, dtype=np.bool_)
    val_r_arr  = np.asarray(v_r_list + [vr_T], dtype=np.float32)
    val_c_arr  = np.asarray(v_c_list + [vc_T], dtype=np.float32)

    return dict(
        obs=obs_arr, next_obs=next_arr,
        act=act_arr, old_logp=logp_arr,
        rew=rew_arr, cost=cost_arr,
        term=term_arr, trunc=trunc_arr,
        val_r=val_r_arr, val_c=val_c_arr,
    )


# ---------------------------------------
# Original CPPO training loop (no dynamics)
# ---------------------------------------
def train_once(
    env_id: str = "SafetyPointGoal1-v0",
    total_iters: int = 1000,
    horizon: int = 1024,              # rollout length T (on-policy)
    gamma: float = 0.99,
    lam: float = 0.95,
    cost_gamma: float = 0.99,
    cost_lam: float = 0.95,
    d_constraint: float = 25.0,       # cost limit (adjust per task spec if needed)
    delta_kl: float = 0.02,           # reverse-KL budget (E-step)
    ratio_lower_bound: float = 0.6,   # lower bound on r_theta (M-step)
    pi_lr: float = 3e-4,
    v_lr: float  = 3e-4,
    seed: int = 0,
    log_dir: str = "./logs_cppo",
):
    """Plain CPPO training loop over `total_iters` iterations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    logger = Logger(log_dir)

    env = safety_gymnasium.make(env_id, render_mode=None)

    # infer dimensions
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    # build networks
    policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim).to(device)
    v_r    = ValueNet(obs_dim=obs_dim).to(device)
    v_c    = CostNet(obs_dim=obs_dim).to(device)

    # optimizers
    pi_opt = optim.Adam(policy.parameters(), lr=pi_lr)
    vr_opt = optim.Adam(v_r.parameters(),    lr=v_lr)
    vc_opt = optim.Adam(v_c.parameters(),    lr=v_lr)

    global_step = 0
    start_time = time.time()
    for it in range(1, total_iters + 1):
        # ---------- (1) collect on-policy rollout ----------
        batch = collect_rollout(env, policy, v_r, v_c, horizon=horizon, device=device)

        # ---------- (2) compute dual GAE (A_r, A_c) & returns ----------
        gae = compute_gae_dual(
            rewards=batch["rew"],
            costs=batch["cost"],
            values_r=batch["val_r"],
            values_c=batch["val_c"],
            terminated=batch["term"],
            truncated=batch["trunc"],
            gamma=gamma, lam=lam,
            cost_gamma=cost_gamma, cost_lam=cost_lam,
            device=device,
        )
        adv_r = gae["adv_r"]   # [T] normalized
        ret_r = gae["ret_r"]   # [T]
        adv_c = gae["adv_c"]   # [T] raw (not normalized)
        ret_c = gae["ret_c"]   # [T]

        # estimate current discounted cost J_c(pi) ~ mean(ret_c) as a proxy
        Jc_pi = float(ret_c.mean().item())

        # ---------- (3) E-step: solve optimal non-parametric ratios v ----------
        est = estep_optimal_v(
            adv_r=adv_r.cpu().numpy(),
            adv_c=adv_c.cpu().numpy(),
            Jc_pi=Jc_pi,
            d_constraint=d_constraint,
            gamma=gamma,
            delta_kl=delta_kl,
            lower_bound=ratio_lower_bound,
        )
        v_target = est["v"]  # numpy [T]

        # ---------- (4) M-step: track v_target with lower-bound clipped ratio ----------
        mstats = mstep_update(
            policy=policy,
            optimizer=pi_opt,
            obs=batch["obs"],
            actions=batch["act"],
            old_logp=batch["old_logp"],
            v_target=v_target,
            clip_lower=ratio_lower_bound,
            entropy_coef=0.0,
            norm_coef=0.01,
            max_grad_norm=0.5,
            device=device,
        )

        # ---------- (5) Critic updates (squared error to returns) ----------
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        ret_r_t = torch.as_tensor(ret_r, dtype=torch.float32, device=device).view(-1, 1)
        ret_c_t = torch.as_tensor(ret_c, dtype=torch.float32, device=device).view(-1, 1)

        vr_pred = v_r(obs_t)
        vc_pred = v_c(obs_t)
        loss_vr = 0.5 * ((vr_pred - ret_r_t) ** 2).mean()
        loss_vc = 0.5 * ((vc_pred - ret_c_t) ** 2).mean()

        vr_opt.zero_grad(set_to_none=True)
        loss_vr.backward()
        torch.nn.utils.clip_grad_norm_(v_r.parameters(), 0.5)
        vr_opt.step()

        vc_opt.zero_grad(set_to_none=True)
        loss_vc.backward()
        torch.nn.utils.clip_grad_norm_(v_c.parameters(), 0.5)
        vc_opt.step()

        # ---------- (6) Logging ----------
        global_step += len(batch["rew"])
        ep_rew = float(np.sum(batch["rew"]))
        ep_cost = float(np.sum(batch["cost"]))
        elapsed = time.time() - start_time
        logger.log(
            episode=it,
            reward=ep_rew,
            cost=ep_cost,
            constraint_violation=max(0.0, ep_cost - d_constraint),
        )
        log_path = os.path.join(log_dir, "train_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[it {it:04d}] step={global_step} | "
                f"EpRew={ep_rew:.1f} EpCost={ep_cost:.1f} | "
                f"Vr={loss_vr.item():.3f} Vc={loss_vc.item():.3f} | "
                f"PiLoss={mstats['loss']:.3f} Track={mstats['track_loss']:.3f} "
                f"MeanRatio={mstats['mean_ratio']:.3f} LB%={mstats['frac_ratio_below_lb']*100:.1f}% | "
                f"time={elapsed/60:.1f}m\n"
            )

    env.close()


# ---------------------------------------
# Dyna-CPPO training loop (with dynamics model)
# ---------------------------------------
def train_dyna_once(
    env_id: str = "SafetyPointGoal1-v0",
    total_iters: int = 1000,
    horizon: int = 1024,
    gamma: float = 0.99,
    lam: float = 0.95,
    cost_gamma: float = 0.99,
    cost_lam: float = 0.95,
    d_constraint: float = 25.0,
    delta_kl: float = 0.02,
    ratio_lower_bound: float = 0.6,
    pi_lr: float = 3e-4,
    v_lr: float  = 3e-4,
    model_lr: float = 1e-3,
    model_horizon: int = 64,            # short-horizon model rollouts
    model_trajs_per_iter: int = 4,      # how many model rollouts per iter
    model_start_iter: int = 10,         # warmup iterations (only real env)
    seed: int = 0,
    log_dir: str = "./logs_dyna",
):
    """Dyna-CPPO training loop with short-horizon model-based rollouts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    logger = Logger(log_dir)

    env = safety_gymnasium.make(env_id, render_mode=None)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim).to(device)
    v_r    = ValueNet(obs_dim=obs_dim).to(device)
    v_c    = CostNet(obs_dim=obs_dim).to(device)

    dyn_model = DynamicsModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=256,
        n_hidden_layers=2,
        predict_reward=True,
        predict_cost=True,
    ).to(device)

    pi_opt  = optim.Adam(policy.parameters(),   lr=pi_lr)
    vr_opt  = optim.Adam(v_r.parameters(),      lr=v_lr)
    vc_opt  = optim.Adam(v_c.parameters(),      lr=v_lr)
    dyn_opt = optim.Adam(dyn_model.parameters(), lr=model_lr)

    global_step = 0
    start_time = time.time()

    for it in range(1, total_iters + 1):
        # ---- (1) Real env rollout ----
        batch_real = collect_rollout(env, policy, v_r, v_c, horizon=horizon, device=device)

        # ---- (2) Dynamics one-step supervised update ----
        obs_t    = torch.as_tensor(batch_real["obs"],      dtype=torch.float32, device=device)
        act_t    = torch.as_tensor(batch_real["act"],      dtype=torch.float32, device=device)
        next_tgt = torch.as_tensor(batch_real["next_obs"], dtype=torch.float32, device=device)
        rew_tgt  = torch.as_tensor(batch_real["rew"],      dtype=torch.float32, device=device).view(-1, 1)
        cost_tgt = torch.as_tensor(batch_real["cost"],     dtype=torch.float32, device=device).view(-1, 1)

        dyn_loss = dyn_model.one_step_loss(
            obs_t, act_t, next_tgt,
            reward_target=rew_tgt,
            cost_target=cost_tgt,
            obs_weight=1.0, reward_weight=1.0, cost_weight=1.0,
        )
        dyn_opt.zero_grad(set_to_none=True)
        dyn_loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn_model.parameters(), 1.0)
        dyn_opt.step()

        # ---- (3) GAE on real data ----
        gae_real = compute_gae_dual(
            rewards=batch_real["rew"],
            costs=batch_real["cost"],
            values_r=batch_real["val_r"],
            values_c=batch_real["val_c"],
            terminated=batch_real["term"],
            truncated=batch_real["trunc"],
            gamma=gamma, lam=lam,
            cost_gamma=cost_gamma, cost_lam=cost_lam,
            device=device,
        )
        adv_r_real = gae_real["adv_r"]
        ret_r_real = gae_real["ret_r"]
        adv_c_real = gae_real["adv_c"]
        ret_c_real = gae_real["ret_c"]

        # ---- (4) Model rollouts & GAE (from real states as seeds) ----
        model_batches = []
        if it >= model_start_iter:
            idx = np.random.randint(0, batch_real["obs"].shape[0], size=model_trajs_per_iter)
            for k in range(model_trajs_per_iter):
                init_obs = batch_real["obs"][idx[k]]
                mb = collect_model_rollout(
                    dyn_model=dyn_model,
                    env_obs_init=init_obs,
                    policy=policy,
                    vr=v_r,
                    vc=v_c,
                    horizon=model_horizon,
                    device=device,
                )
                model_batches.append(mb)

        if len(model_batches) > 0:
            def concat(key: str) -> np.ndarray:
                return np.concatenate([b[key] for b in model_batches], axis=0)

            batch_model = dict(
                obs      = concat("obs"),
                next_obs = concat("next_obs"),
                act      = concat("act"),
                old_logp = concat("old_logp"),
                rew      = concat("rew"),
                cost     = concat("cost"),
                term     = concat("term"),
                trunc    = concat("trunc"),
            )
            # value heads for model data
            with torch.no_grad():
                obs_m = torch.as_tensor(batch_model["obs"], dtype=torch.float32, device=device)
                val_r_m = v_r(obs_m).squeeze(-1).cpu().numpy()
                val_c_m = v_c(obs_m).squeeze(-1).cpu().numpy()

                last_obs_m = torch.as_tensor(
                    batch_model["next_obs"][-1], dtype=torch.float32, device=device
                ).unsqueeze(0)
                val_r_T = v_r(last_obs_m).item()
                val_c_T = v_c(last_obs_m).item()

            val_r_m = np.concatenate([val_r_m, [val_r_T]], axis=0)
            val_c_m = np.concatenate([val_c_m, [val_c_T]], axis=0)

            gae_model = compute_gae_dual(
                rewards=batch_model["rew"],
                costs=batch_model["cost"],
                values_r=val_r_m,
                values_c=val_c_m,
                terminated=batch_model["term"],
                truncated=batch_model["trunc"],
                gamma=gamma, lam=lam,
                cost_gamma=cost_gamma, cost_lam=cost_lam,
                device=device,
            )
            adv_r_model = gae_model["adv_r"]
            ret_r_model = gae_model["ret_r"]
            adv_c_model = gae_model["adv_c"]
            ret_c_model = gae_model["ret_c"]

            # combine real + model data
            obs_all    = np.concatenate([batch_real["obs"], batch_model["obs"]], axis=0)
            act_all    = np.concatenate([batch_real["act"], batch_model["act"]], axis=0)
            oldlog_all = np.concatenate([batch_real["old_logp"], batch_model["old_logp"]], axis=0)

            adv_r_all  = torch.cat([adv_r_real, adv_r_model], dim=0)
            ret_r_all  = torch.cat([ret_r_real, ret_r_model], dim=0)
            adv_c_all  = torch.cat([adv_c_real, adv_c_model], dim=0)
            ret_c_all  = torch.cat([ret_c_real, ret_c_model], dim=0)
        else:
            obs_all    = batch_real["obs"]
            act_all    = batch_real["act"]
            oldlog_all = batch_real["old_logp"]
            adv_r_all  = adv_r_real
            ret_r_all  = ret_r_real
            adv_c_all  = adv_c_real
            ret_c_all  = ret_c_real

        # ---- (5) E-step with combined data ----
        Jc_pi = float(ret_c_all.mean().item())

        est = estep_optimal_v(
            adv_r=adv_r_all.cpu().numpy(),
            adv_c=adv_c_all.cpu().numpy(),
            Jc_pi=Jc_pi,
            d_constraint=d_constraint,
            gamma=gamma,
            delta_kl=delta_kl,
            lower_bound=ratio_lower_bound,
        )
        v_target = est["v"]

        # ---- (6) M-step policy update ----
        mstats = mstep_update(
            policy=policy,
            optimizer=pi_opt,
            obs=obs_all,
            actions=act_all,
            old_logp=oldlog_all,
            v_target=v_target,
            clip_lower=ratio_lower_bound,
            entropy_coef=0.0,
            norm_coef=0.01,
            max_grad_norm=0.5,
            device=device,
        )

        # ---- (7) Critic update with combined data ----
        obs_t_all   = torch.as_tensor(obs_all,   dtype=torch.float32, device=device)
        ret_r_t_all = torch.as_tensor(ret_r_all, dtype=torch.float32, device=device).view(-1, 1)
        ret_c_t_all = torch.as_tensor(ret_c_all, dtype=torch.float32, device=device).view(-1, 1)

        vr_pred = v_r(obs_t_all)
        vc_pred = v_c(obs_t_all)
        loss_vr = 0.5 * ((vr_pred - ret_r_t_all) ** 2).mean()
        loss_vc = 0.5 * ((vc_pred - ret_c_t_all) ** 2).mean()

        vr_opt.zero_grad(set_to_none=True)
        loss_vr.backward()
        torch.nn.utils.clip_grad_norm_(v_r.parameters(), 0.5)
        vr_opt.step()

        vc_opt.zero_grad(set_to_none=True)
        loss_vc.backward()
        torch.nn.utils.clip_grad_norm_(v_c.parameters(), 0.5)
        vc_opt.step()

        # ---- (8) Logging (still log real env episode rew/cost) ----
        global_step += len(batch_real["rew"])
        ep_rew  = float(np.sum(batch_real["rew"]))
        ep_cost = float(np.sum(batch_real["cost"]))
        elapsed = time.time() - start_time

        logger.log(
            episode=it,
            reward=ep_rew,
            cost=ep_cost,
            constraint_violation=max(0.0, ep_cost - d_constraint),
        )
        log_path = os.path.join(log_dir, "train_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[it {it:04d}] step={global_step} | "
                f"EpRew={ep_rew:.1f} EpCost={ep_cost:.1f} | "
                f"Vr={loss_vr.item():.3f} Vc={loss_vc.item():.3f} | "
                f"PiLoss={mstats['loss']:.3f} Track={mstats['track_loss']:.3f} "
                f"MeanRatio={mstats['mean_ratio']:.3f} LB%={mstats['frac_ratio_below_lb']*100:.1f}% | "
                f"DynLoss={dyn_loss.item():.3f} | "
                f"time={elapsed/60:.1f}m\n"
            )

    env.close()


if __name__ == "__main__":
    # train_once(
    #     env_id="SafetyPointGoal1-v0",
    #     total_iters=10000,
    #     horizon=1024,
    #     gamma=0.99, lam=0.95,
    #     cost_gamma=0.99, cost_lam=0.95,
    #     d_constraint=25.0,
    #     delta_kl=0.02,
    #     ratio_lower_bound=0.6,
    #     pi_lr=3e-4,
    #     v_lr=3e-4,
    #     seed=0,
    #     log_dir="./logs",
    # )

    train_dyna_once(
        env_id="SafetyPointGoal1-v0",
        total_iters=10000,
        horizon=1024,
        gamma=0.99, lam=0.95,
        cost_gamma=0.99, cost_lam=0.95,
        d_constraint=25.0,
        delta_kl=0.02,
        ratio_lower_bound=0.6,
        pi_lr=3e-4,
        v_lr=3e-4,
        model_lr=1e-3,
        model_horizon=64,
        model_trajs_per_iter=4,
        model_start_iter=10,
        seed=0,
        log_dir="./logs",
    )

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ===========================================================
# Logger
# ===========================================================

@dataclass
class PPOLogger:
    """
    Accumulates per-update scalars during a training run
    """

    updates:     List[int]   = field(default_factory=list)
    steps:       List[int]   = field(default_factory=list)
    rewards:     List[float] = field(default_factory=list)
    reward_steps:List[int]   = field(default_factory=list)
    ev:          List[float] = field(default_factory=list)
    vloss:       List[float] = field(default_factory=list)
    ploss:       List[float] = field(default_factory=list)
    entropy:     List[float] = field(default_factory=list)
    kl:          List[float] = field(default_factory=list)
    lr:          List[float] = field(default_factory=list)
    sps:         List[float] = field(default_factory=list)

    def log(
        self,
        update:      int,
        global_step: int,
        ev:          float,
        vloss:       float,
        ploss:       float,
        entropy:     float,
        kl:          float,
        lr:          float,
        sps:         float,
        reward:      Optional[float] = None,
    ) -> None:
        
        self.updates.append(update)
        self.steps.append(global_step)
        self.ev.append(ev)
        self.vloss.append(vloss)
        self.ploss.append(ploss)
        self.entropy.append(entropy)
        self.kl.append(kl)
        self.lr.append(lr)
        self.sps.append(sps)

        if reward is not None:
            self.rewards.append(reward)
            self.reward_steps.append(global_step)

    def smooth(self, values: List[float], window: int = 10) -> np.ndarray:
        """
        Simple moving average, edge-padded
        """

        arr = np.array(values, dtype=float)

        if len(arr) < window:
            return arr
        
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")
    
# ===========================================================
# Style helpers
# ===========================================================

PALETTE = {
    "reward":  "#3266ad",
    "ev":      "#1D9E75",
    "vloss":   "#D85A30",
    "ploss":   "#7F77DD",
    "entropy": "#BA7517",
    "kl":      "#73726c",
    "lr":      "#888780",
    "sps":     "#533ab7",
}

def _style_ax(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "global step") -> None:
    ax.set_title(title, fontsize=11, fontweight="normal", pad=6)
    ax.set_ylabel(ylabel, fontsize=9, color="#555")
    ax.set_xlabel(xlabel, fontsize=9, color="#555")
    ax.tick_params(labelsize=8, colors="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ccc")
    ax.grid(axis="y", color="#eee", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)


def _thousands(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )


# ===========================================================
# Main plot function
# ===========================================================

def plot_training(
    logger:      PPOLogger,
    smooth_window: int  = 10,
    figsize:     tuple  = (14, 10),
    target_kl:   Optional[float] = None,
    save_path:   Optional[str]   = None,
) -> plt.Figure:
    """
    Render a 3 by 3 dashboard of PPO training diagnostics.

    Parameters
    ----------
    logger        : PPOLogger instance populated during training.
    smooth_window : Window size for the moving-average overlay (0 to disable).
    figsize       : Overall figure size in inches.
    target_kl     : If set, draws a dashed reference line on the KL plot.
    save_path     : If set, saves the figure to this path (e.g. 'training.png').
    """
    steps = np.array(logger.steps)

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle("PPO training diagnostics", fontsize=13, fontweight="normal", y=1.01)
    plt.subplots_adjust(hspace=0.55, wspace=0.35)

    def _line(ax, x, y, color, label=None, alpha_raw=0.25):
        y = np.array(y, dtype=float)
        ax.plot(x, y, color=color, alpha=alpha_raw, linewidth=0.8)

        if smooth_window > 1 and len(y) >= smooth_window:
            ax.plot(x, logger.smooth(y, smooth_window), color=color, linewidth=1.8, label=label)
        else:
            ax.lines[-1].set_alpha(1.0)

    # Episodeic Return
    ax = axes[0, 0]
    if logger.rewards:
        rx = np.array(logger.reward_steps)
        ry = np.array(logger.rewards)
        ax.scatter(rx, ry, s=8, color=PALETTE["reward"], alpha=0.35, zorder=2)

        if len(ry) >= smooth_window:
            ax.plot(rx, logger.smooth(ry, smooth_window), color=PALETTE["reward"], linewidth=1.8)
            
    _style_ax(ax, "episodic return", "reward")
    _thousands(ax)

    # Explained Variance
    ax = axes[0, 1]
    _line(ax, steps, logger.ev, PALETTE["ev"])
    ax.axhline(1.0, color="#bbb", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="#bbb", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.1, 1.05)
    _style_ax(ax, "explained variance", "EV  ->  1 = perfect")
    _thousands(ax)

    # Value Loss
    ax = axes[0, 2]
    _line(ax, steps, logger.vloss, PALETTE["vloss"])
    _style_ax(ax, "value loss", "loss  ↓")
    _thousands(ax)

    # Policy Loss
    ax = axes[1, 0]
    _line(ax, steps, logger.ploss, PALETTE["ploss"])
    ax.axhline(0.0, color="#bbb", linewidth=0.8, linestyle="--")
    _style_ax(ax, "policy (surrogate) loss", "loss")
    _thousands(ax)

    # Entropy
    ax = axes[1, 1]
    _line(ax, steps, logger.entropy, PALETTE["entropy"])
    _style_ax(ax, "entropy  (exploration)", "entropy  ↓ over time")
    _thousands(ax)

    # Approx KL Divergence
    ax = axes[1, 2]
    _line(ax, steps, logger.kl, PALETTE["kl"])
    if target_kl is not None:
        ax.axhline(target_kl, color="#e24b4a", linewidth=1.0,
                   linestyle="--", label=f"target KL = {target_kl}")
        ax.legend(fontsize=8, frameon=False)
    _style_ax(ax, "approx KL divergence", "KL  (stay below target)")
    _thousands(ax)

    # Learning Rate (Constant; did not use annealing for final model)
    ax = axes[2, 0]
    ax.plot(steps, logger.lr, color=PALETTE["lr"], linewidth=1.5)
    _style_ax(ax, "learning rate schedule", "lr")
    _thousands(ax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))

    # Steps/sec
    ax = axes[2, 1]
    _line(ax, steps, logger.sps, PALETTE["sps"])
    _style_ax(ax, "throughput", "steps / sec")
    _thousands(ax)

    # Rewards vs EV
    ax = axes[2, 2]
    if logger.rewards and logger.ev:
        # align EV to reward timestamps by nearest-update index
        ev_arr  = np.array(logger.ev)
        rx_idx  = [np.argmin(np.abs(steps - s)) for s in logger.reward_steps]
        ev_at_r = ev_arr[rx_idx]
        ry      = np.array(logger.rewards)
        sc = ax.scatter(ev_at_r, ry, s=14, c=logger.reward_steps,
                        cmap="Blues", alpha=0.7, zorder=2)
        fig.colorbar(sc, ax=ax, label="global step", pad=0.02)
    _style_ax(ax, "reward vs EV  (colour = time)", "reward", xlabel="explained variance")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved -> {save_path}")

    return fig


# ===========================================================
# Convenience: plot from TensorBoard event files
# ===========================================================

def plot_from_tensorboard(log_dir: str, **kwargs) -> plt.Figure:
    """
    Load scalars directly from a TensorBoard runs/ directory and plot them.

    Requires:  pip install tensorboard
    Example:   plot_from_tensorboard("runs/CartPole-v1__vectorized_arch__1__...")
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("pip install tensorboard  — needed to read event files.")

    ea = EventAccumulator(log_dir)
    ea.Reload()

    def _get(tag):
        try:
            events = ea.Scalars(tag)
            steps  = [e.step  for e in events]
            vals   = [e.value for e in events]
            return steps, vals
        except KeyError:
            return [], []

    logger = PPOLogger()

    r_steps, r_vals   = _get("charts/episodic_return")
    ev_steps, ev_vals = _get("losses/explained_variance")
    vl_steps, vl_vals = _get("losses/value_loss")
    pl_steps, pl_vals = _get("losses/policy_loss")
    en_steps, en_vals = _get("losses/entropy")
    kl_steps, kl_vals = _get("losses/approx_kl")
    lr_steps, lr_vals = _get("charts/learning_rate")
    sp_steps, sp_vals = _get("charts/SPS")

    # align everything to the longest shared step sequence
    ref_steps = ev_steps or vl_steps or lr_steps
    for i, s in enumerate(ref_steps):
        def _at(sts, vs):
            if not sts:
                return float("nan")
            idx = min(range(len(sts)), key=lambda j: abs(sts[j]-s))
            return vs[idx]
        
        ep_return = None
        if "episode" in info:
            ep_return = float(info["episode"]["r"][0])

        logger.log(
            update      = i + 1,
            global_step = s,
            reward      = ep_return,
            ev          = _at(ev_steps, ev_vals),
            vloss       = _at(vl_steps, vl_vals),
            ploss       = _at(pl_steps, pl_vals),
            entropy     = _at(en_steps, en_vals),
            kl          = _at(kl_steps, kl_vals),
            lr          = _at(lr_steps, lr_vals),
            sps         = _at(sp_steps, sp_vals),
        )

    for s, v in zip(r_steps, r_vals):
        logger.rewards.append(v)
        logger.reward_steps.append(s)

    return plot_training(logger, **kwargs)
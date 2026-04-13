# %% Phase-plane / nullcline analysis of the CGM–SF feedback loop


import importlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy import cosmology
import astropy.units as u
import astropy.constants as consts
from scipy.optimize import brentq

import cgm_sf_regulator
from cgm_sf_regulator import CGMRegulator, mhalo_at_z0_fakhouri

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)
H0 = 70
LCDM = cosmology.LambdaCDM(H0=H0, Om0=0.3, Ode0=0.7)

# %% run a model for a MW mass halo
mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 13.3)  # Gyr

model = CGMRegulator(
    mhalo_z0,
    t_span,
    tstep=0.005,
    add_f_prevent_floor=1e-8,  # virtually no floor
    KS_kappa_s=0.02,
    KS_n=1.8,
    disk_scale_length=0.018,
    KS_parametrization="KS1998",
    TEST_tej_Tvir_definition=False,
    eta_z=0.7,
)
model.run_halo()
res = model.get_results()
der = model.get_derived_quantities()


# %% ── 2.  Evaluate reduced 2-D vector field ────────────────────────────
def phase_plane_field(
    model,
    t_ref,
    res,
    m_cgm_hot_grid,
    e_cgm_grid,
    freeze_cold_from_hot_ratio=None,
):
    """Compute (dM_cgm_hot/dt, dE_cgm/dt) on a 2-D grid.

    other variables (M_ism, M_star, M_bulge, M_halo, metals) are
    frozen at their solved values at *t_ref*.  M_cgm_cold is either
    frozen or slaved to M_cgm_hot via a fixed ratio.

    parameters
    ----------
    model : CGMRegulator (already solved)
    t_ref : float    reference time [Gyr]
    res   : dict     model.get_results()
    m_cgm_hot_grid : 1-D array [Msun]
    e_cgm_grid     : 1-D array [erg]
    freeze_cold_from_hot_ratio : float or None
        If None, M_cgm_cold is frozen at its solved value at t_ref.
        If a float, M_cgm_cold = ratio * M_cgm_hot (slaved).

    returns
    -------
    Mh, Eg : 2-D meshgrids of the input variables
    dMh, dEg : 2-D arrays of the derivatives at each grid point
    derived_grid : dict of 2-D arrays of selected derived quantities
    """
    t_arr = np.asarray(res["t"])
    idx = int(np.argmin(np.abs(t_arr - t_ref)))

    # frozen background state at t_ref
    m_ism_0 = res["m_ism"][idx]
    m_star_0 = res["m_star"][idx]
    m_bulge_0 = res["m_bulge"][idx]
    m_cgm_cold_0 = res["m_cgm_cold"][idx]
    m_metals_cgm_0 = res["m_metals_cgm"][idx]
    m_metals_ism_0 = res["m_metals_ism"][idx]
    m_halo_0 = res["m_halo"][idx]

    Mh, Eg = np.meshgrid(m_cgm_hot_grid, e_cgm_grid, indexing="ij")
    dMh = np.zeros_like(Mh)
    dEg = np.zeros_like(Eg)

    # derived quantities we want to record
    tcool_grid = np.zeros_like(Mh)
    fprevent_grid = np.zeros_like(Mh)
    Tcgm_grid = np.zeros_like(Mh)
    sfr_grid = np.zeros_like(Mh)
    dot_m_cool_grid = np.zeros_like(Mh)

    ny, nx = Mh.shape

    for i in range(ny):
        for j in range(nx):
            m_hot_ij = float(Mh[i, j])
            e_cgm_ij = float(Eg[i, j])

            if freeze_cold_from_hot_ratio is not None:
                m_cold_ij = freeze_cold_from_hot_ratio * m_hot_ij
            else:
                m_cold_ij = m_cgm_cold_0

            state = np.array(
                [
                    m_ism_0,
                    m_star_0,
                    m_bulge_0,
                    m_hot_ij,
                    m_cold_ij,
                    m_metals_cgm_0,
                    m_metals_ism_0,
                    m_halo_0,
                    e_cgm_ij,
                ]
            )
            # ODE derivatives (9-vector)
            derivs = model.mass_evolution(float(t_ref), state, ode_mode=True)
            dMh[i, j] = derivs[3]  # dot_m_cgm_hot
            dEg[i, j] = derivs[8]  # dot_e_cgm
            # derived quantities
            derived_quantitites = model.mass_evolution(
                float(t_ref), state, ode_mode=False
            )
            tcool_grid[i, j] = derived_quantitites[13]  # tcool_real
            fprevent_grid[i, j] = derived_quantitites[10]  # f_prevent
            Tcgm_grid[i, j] = derived_quantitites[19]  # T_cgm
            sfr_grid[i, j] = derived_quantitites[16]  # sfr
            dot_m_cool_grid[i, j] = derived_quantitites[6]  # dot_m_cgm_hot

    derived_grid = dict(
        tcool=tcool_grid,
        f_prevent=fprevent_grid,
        T_cgm=Tcgm_grid,
        sfr=sfr_grid,
        dot_m_cool=dot_m_cool_grid,
    )
    return Mh, Eg, dMh, dEg, derived_grid


def plot_phase_plane(
    Mh,
    Eg,
    dMh,
    dEg,
    derived_grid,
    res,
    t_ref,
    title_extra="",
    ax=None,
    show_trajectory=True,
    plot=True,
    model=None,
):
    """
    phase-portrait plot

    parameters
    ----------
    plot : bool
        If False, only compute and return fixed-point data without plotting.
    """
    if model is None:
        model = globals().get("model", None)
    if model is None:
        raise ValueError("A CGMRegulator model must be provided via model=...")

    if plot and ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    else:
        fig = ax.figure if ax is not None else None

    # work in log10 space so streamplot gets linearly-spaced axes
    # chain rule:  d(log10 X)/dt = dX/dt / (X * ln10)
    ln10 = np.log(10.0)
    log_Mh = np.log10(Mh)
    log_Eg = np.log10(Eg)
    dlog_Mh = dMh / (Mh * ln10)  # Gyr^-1
    dlog_Eg = dEg / (Eg * ln10)  # Gyr^-1

    # magnitude for colour (in log phase-space units)
    magnitude = np.sqrt(dlog_Mh**2 + dlog_Eg**2)
    pad_episilon = 1e-30
    log_magnitude = np.log10(magnitude + pad_episilon)
    selection_metric_log = np.log10(np.hypot(dMh, dEg) + pad_episilon)

    # streamplot axes must be 1-D and equally spaced
    x_vals = log_Mh[:, 0]
    y_vals = log_Eg[0, :]
    x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))

    if plot and ax is not None:
        strm = ax.streamplot(
            x_vals,
            y_vals,
            dlog_Mh.T,
            dlog_Eg.T,
            color=log_magnitude,
            cmap="jet",
            linewidth=0.8,
            density=3,
            arrowsize=0.8,
        )

        cbar = fig.colorbar(strm.lines, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label(
            r"$\log  \sqrt{(d\log  M_{\rm CGM,hot}/dt)^2 + (d\log  E_{\rm CGM}/dt)^2}$ [Gyr$^{-1}$]"
        )

        # nullclines
        ax.contour(
            log_Mh.T,
            log_Eg.T,
            dMh.T,
            levels=[0],
            colors="dodgerblue",
            linewidths=2.5,
            linestyles="-",
        )
        ax.contour(
            log_Mh.T,
            log_Eg.T,
            dEg.T,
            levels=[0],
            colors="tomato",
            linewidths=2.5,
            linestyles="-",
        )

        # dummy artists for legend
        ax.plot([], [], color="dodgerblue", lw=2.5, label=r"$\dot{M}_{\rm CGM,hot}=0$")
        ax.plot([], [], color="tomato", lw=2.5, label=r"$\dot{E}_{\rm CGM}=0$")

        # overlay trajectory
        if show_trajectory:
            t_traj = np.asarray(res["t"])
            traj_Eg = np.asarray(res["egy_cgm"])
            traj_Mh = np.asarray(res["m_cgm_hot"])
            pos_mask = (traj_Eg > 0) & (traj_Mh > 0)
            traj_log_Eg = np.log10(traj_Eg[pos_mask])
            traj_log_Mh = np.log10(traj_Mh[pos_mask])

            ax.plot(
                traj_log_Mh,
                traj_log_Eg,
                color="grey",
                lw=2,
                ls="-",
                alpha=0.8,
                label="trajectory (start to end)",
            )

            start_x, start_y = traj_log_Mh[0], traj_log_Eg[0]
            end_x, end_y = traj_log_Mh[-1], traj_log_Eg[-1]
            if (x_min <= start_x <= x_max) and (y_min <= start_y <= y_max):
                ax.plot(
                    start_x,
                    start_y,
                    "^",
                    color="white",
                    markeredgecolor="k",
                    ms=7,
                    zorder=10,
                    label="start",
                )
            if (x_min <= end_x <= x_max) and (y_min <= end_y <= y_max):
                ax.plot(
                    end_x,
                    end_y,
                    "s",
                    color="black",
                    markeredgecolor="white",
                    ms=6,
                    zorder=10,
                    label="end",
                )

            idx = int(np.argmin(np.abs(t_traj - t_ref)))
            ref_log_Eg = np.log10(res["egy_cgm"][idx])
            ref_log_Mh = np.log10(res["m_cgm_hot"][idx])
            if (x_min <= ref_log_Mh <= x_max) and (y_min <= ref_log_Eg <= y_max):
                ax.plot(
                    ref_log_Mh,
                    ref_log_Eg,
                    "o",
                    color="lime",
                    markeredgecolor="k",
                    ms=8,
                    zorder=10,
                    label=f"current time (t={t_ref:.2f} Gyr)",
                )

    # this is a bit hacky but we want to find where the nullclines cross (fixed point candidates), and for that we need to know the signs of dMh and dEg at the corners of each cell in the grid.  So we create 4 shifted versions of dMh and dEg to get the values at the corners of each cell.
    dMh_00, dMh_10 = dMh[:-1, :-1], dMh[1:, :-1]
    dMh_01, dMh_11 = dMh[:-1, 1:], dMh[1:, 1:]
    dEg_00, dEg_10 = dEg[:-1, :-1], dEg[1:, :-1]
    dEg_01, dEg_11 = dEg[:-1, 1:], dEg[1:, 1:]

    dMh_min = np.minimum.reduce([dMh_00, dMh_10, dMh_01, dMh_11])
    dMh_max = np.maximum.reduce([dMh_00, dMh_10, dMh_01, dMh_11])
    dEg_min = np.minimum.reduce([dEg_00, dEg_10, dEg_01, dEg_11])
    dEg_max = np.maximum.reduce([dEg_00, dEg_10, dEg_01, dEg_11])

    cross_Mh = (
        np.isfinite(dMh_min)
        & np.isfinite(dMh_max)
        & (dMh_min <= 0.0)
        & (dMh_max >= 0.0)
    )
    cross_Eg = (
        np.isfinite(dEg_min)
        & np.isfinite(dEg_max)
        & (dEg_min <= 0.0)
        & (dEg_max >= 0.0)
    )
    both = cross_Mh & cross_Eg

    if np.any(both):
        # if there are any cells where both dMh and dEg change sign, we consider those as candidate fixed points.  We compute a "score" for each cell based on the magnitudes of dMh and dEg at the corners, and pick the cell with the smallest score as our best estimate for the fixed point location.  This is a heuristic approach to find the most likely fixed point when we have a discrete grid, but it should work reasonably well if the grid is fine enough.
        cell_score = 0.25 * (
            np.hypot(dMh_00, dEg_00)
            + np.hypot(dMh_10, dEg_10)
            + np.hypot(dMh_01, dEg_01)
            + np.hypot(dMh_11, dEg_11)
        )
        masked_score = np.where(both, cell_score, np.inf)
        i, j = np.unravel_index(np.argmin(masked_score), masked_score.shape)
        Mh_fp = np.sqrt(Mh[i, j] * Mh[i + 1, j + 1])
        Eg_fp = np.sqrt(Eg[i, j] * Eg[i + 1, j + 1])
        fp_reason = "nullcline-crossing cell score"
        fp_value = float(masked_score[i, j])
    else:
        # if no exact nullcline crossing, just pick the point with the smallest flow magnitude as a "best-effort" fixed point for local stability analysis.  This is not guaranteed to be a true fixed point, but it can still provide insight into the local flow structure.
        flow_norm = np.hypot(dMh, dEg)
        i, j = np.unravel_index(np.nanargmin(flow_norm), flow_norm.shape)
        Mh_fp = Mh[i, j]
        Eg_fp = Eg[i, j]
        fp_reason = "minimum flow magnitude"
        fp_value = float(flow_norm[i, j])

    # local Jacobian via finite differences
    # the current time is use to build the vector field
    # the current state is used to set the frozen background variables and to choose appropriate finite difference step sizes for the fast variables
    idx_ref = int(np.argmin(np.abs(np.asarray(res["t"]) - t_ref)))
    state_fp = np.array(
        [
            res["m_ism"][idx_ref],
            res["m_star"][idx_ref],
            res["m_bulge"][idx_ref],
            Mh_fp,
            res["m_cgm_cold"][idx_ref],
            res["m_metals_cgm"][idx_ref],
            res["m_metals_ism"][idx_ref],
            res["m_halo"][idx_ref],
            Eg_fp,
        ]
    )
    # choose finite difference step sizes adaptively based on the scale of the fixed point
    deltaM = max(abs(Mh_fp) * 1e-5, 1e2) # at least 100 Msun to avoid numerical issues near zero
    deltaE = max(abs(Eg_fp) * 1e-5, 1e36) # at least 1e36 erg to avoid numerical issues near zero

    def eval_2d(dm, de):
        s = state_fp.copy()
        s[3] += dm # perturb M_cgm_hot
        s[8] += de # perturb E_cgm
        d = model.mass_evolution(float(t_ref), s, ode_mode=True)
        return d[3], d[8]


    J = np.zeros((2, 2))
    fp, fm = eval_2d(deltaM, 0), eval_2d(-deltaM, 0) # perturb only M_cgm_hot
    J[0, 0] = (fp[0] - fm[0]) / (2 * deltaM) # d(dM_cgm_hot)/dM_cgm_hot
    J[1, 0] = (fp[1] - fm[1]) / (2 * deltaM) # d(dE_cgm)/dM_cgm_hot
    fp, fm = eval_2d(0, deltaE), eval_2d(0, -deltaE) # perturb only E_cgm
    J[0, 1] = (fp[0] - fm[0]) / (2 * deltaE)  # d(dM_cgm_hot)/dE_cgm
    J[1, 1] = (fp[1] - fm[1]) / (2 * deltaE) # d(dE_cgm)/dE_cgm

    eigvals = np.linalg.eigvals(J)
    tr = np.real(eigvals[0]) + np.real(eigvals[1])
    det = np.real(eigvals[0] * eigvals[1])
    disc = tr**2 - 4 * det

    if det < 0:
        fp_type = "saddle"
    elif disc < 0:
        fp_type = "stable spiral" if tr < 0 else "unstable spiral"
    else:
        fp_type = "stable node" if tr < 0 else "unstable node"

    print("[phase-plane] reason:", fp_reason)
    print(
        "[phase-plane] selected point:",
        f"log10(Mh)={np.log10(Mh_fp):.4f}, log10(Eg)={np.log10(Eg_fp):.4f}, t={t_ref:.3f} Gyr, value={fp_value:.6e}",
    )
    print(
        "[phase-plane] classification:",
        fp_type,
        f"| eigvals={eigvals}",
        f"| tr={tr:.6e}",
        f"| det={det:.6e}",
        f"| disc={disc:.6e}",
    )
    print("[phase-plane] Jacobian:\n", J)

    if plot and ax is not None:
        ax.plot(
            np.log10(Mh_fp),
            np.log10(Eg_fp),
            "*",
            color="yellow",
            ms=14,
            markeredgecolor="k",
            zorder=11,
            label="fixed point",
        )

        z_ref = cosmology.z_at_value(LCDM.age, t_ref * u.Gyr)
        ax.set(
            xlabel=r"$\log (M_{\rm CGM,hot}$ / ${\rm M_\odot})$",
            ylabel=r"$\log (E_{\rm CGM}$ / erg)",
            title=f"Phase plane at $t={t_ref:.2f}$ Gyr  ($z={z_ref:.2f}$)"
            + title_extra,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(frameon=False, fontsize=10, loc="lower right")

    return dict(
        eigvals=eigvals,  # for fixed-point classification
        tr=tr,
        det=det,
        disc=disc,
        fp_type=fp_type,  # e.g. "stable spiral"
        Mh_fp=Mh_fp,  # fixed-point coordinates
        Eg_fp=Eg_fp,  # fixed-point coordinates
        fp_reason=fp_reason,
        fp_value=fp_value,
        jacobian_2x2=J,  # local Jacobian matrix at the fixed point
        x_vals=x_vals,  # for streamplot, x = log10(Mh)
        y_vals=y_vals,  # for streamplot, y = log10(Eg)
        log_Mh=log_Mh,  # for contour plotting, log10(Mh)
        log_Eg=log_Eg,  # for contour plotting, log10(Eg)
        dlog_Mh=dlog_Mh,  # for contour plotting, d(log10(Mh))/dt
        dlog_Eg=dlog_Eg,  # for contour plotting, d(log10(Eg))/dt
        color=log_magnitude,
        selection_metric_log=selection_metric_log,
    )


# %% plot only mass/energy rate evolution (no log M_cgm_hot or log E_cgm panels)
fig, ax = plt.subplots(
    2,
    1,
    figsize=(4, 6),
    dpi=200,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]},
)
plt.subplots_adjust(hspace=0.11, wspace=0.0)

red1 = "tab:red"
red2 = "tab:orange"
blu1 = "dodgerblue"
blu2 = "tab:green"

tmax = res["t"][-1]

# Panel (0,0): M_CGM mass rate breakdown
dot_m_cgm_cooling = res["m_cgm_hot"] / der["tcool_real"]
dot_m_cgm_in = der["dot_m_cgm_in"]
dot_m_sne_wind = der["dot_m_ism_wind"]
dot_m_cgm_ej = der["dot_m_cgm_out"]
dot_cgm_falling = res["m_cgm_cold"] / der["t_dynamical"]


ax[0].plot(
    res["t"],
    np.log10(np.abs(dot_m_cgm_in)),
    lw=2,
    label=r"$\dot{M}_{\rm in}$",
    color=red1,
)
ax[0].plot(
    res["t"],
    np.log10(np.abs(dot_m_sne_wind)),
    lw=2,
    label=r"$\dot{M}_{\rm SNe, wind}$",
    color=red2,
)
ax[0].plot(
    res["t"],
    np.log10(np.abs(dot_m_cgm_cooling)),
    lw=2,
    label=r"$\dot{M}_{\rm  cooling}$",
    color=blu1,
)
ax[0].plot(
    res["t"],
    np.log10(np.abs(dot_m_cgm_ej)),
    lw=2,
    label=r"$\dot{M}_{\rm ej}$",
    color=blu2,
    ls="--",
)
# plot the mdot hot total
dot_m_cgm_hot_total = dot_m_cgm_in + dot_m_sne_wind - dot_m_cgm_cooling - dot_m_cgm_ej
ax[0].plot(
    res["t"],
    np.log10(np.abs(dot_m_cgm_hot_total)),
    lw=2,
    # label=r"total$\dot{M}_{\rm CGM, hot}$ (negative)",
    color="k",
    ls=":",
    alpha=0.8,
)
ax[0].plot(
    res["t"],
    np.log10(dot_m_cgm_hot_total),
    lw=2,
    label=r"total $\dot{M}_{\rm CGM, hot}$",
    color="k",
    ls="-",
    alpha=0.8,
)
ax[0].set(
    ylabel=r"$\log$ mass rates [M$_{\odot}$ Gyr$^{-1}$]", yscale="linear", ylim=(5.5, 10.5)
)
ax[0].legend(frameon=False, ncol=1, fontsize=11, loc="lower right")
ax[0].grid(True, alpha=0.3, which="both")

# Panel (1,0): E_CGM energy rate breakdown
dot_e_cgm_acc = der["dot_e_cgm_in"]
dot_e_sne_wind = der["dot_e_ism_wind"]
dot_e_cgm_cool = der["dot_e_cgm_cooling"]
dot_e_cgm_ej = der["dot_e_cgm_out"]

ax[1].plot(
    res["t"],
    np.log10(np.abs(dot_e_cgm_acc)),
    color=red1,
    lw=2,
    label=r"$\dot{E}_{\rm in}$",
)
ax[1].plot(
    res["t"],
    np.log10(np.abs(dot_e_sne_wind)),
    color=red2,
    lw=2,
    label=r"$\dot{E}_{\rm SNe, wind}$",
)
ax[1].plot(
    res["t"],
    np.log10(np.abs(dot_e_cgm_cool)),
    color=blu1,
    lw=2,
    label=r"$\dot{E}_{\rm cooling}$",
    ls="--",
)
ax[1].plot(
    res["t"],
    np.log10(np.abs(dot_e_cgm_ej)),
    color=blu2,
    lw=2,
    label=r"$\dot{E}_{\rm ej}$",
    ls="--",
)
# plot the totals
dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool
ax[1].plot(
    res["t"],
    np.log10(np.abs(dot_e_cgm_total)),
    color="k",
    lw=2,
    # label=r"total $\dot{E}_{\rm CGM}$ (negative)",
    ls=":",
    alpha=0.8,
)
ax[1].plot(
    res["t"],
    np.log10(dot_e_cgm_total),
    color="k",
    lw=2,
    label=r"total $\dot{E}_{\rm CGM}$",
    ls="-",
    alpha=0.8,
)

ax[1].set(
    ylabel=r"$\log$ energy rates [erg Gyr$^{-1}$]",
    yscale="linear",
    xscale="log",
    xlabel=r"$t_{\rm univ}$ [Gyr]",
    xlim=(0.1, tmax),
    ylim=(53, 58),
)
ax[1].legend(frameon=False, ncol=1, fontsize=11, loc="lower right")
ax[1].grid(True, alpha=0.3, which="both")

# Add redshift as a twin axis on top
ax_z = ax[0].twiny()
ax_z.set_xscale("log")
ax_z.set_xlim(ax[0].get_xlim())
# make a twin redshift axis for the top row, using z
t_ticks = np.array([0.1, 0.3, 1, 3, 8, 13.3])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
ax_z.set_xticks(t_ticks)
ax_z.set_xticklabels([f"{z:.1f}" for z in z_ticks])
ax_z.set_xlabel(r"$z$")
z_start = cosmology.z_at_value(LCDM.age, res["t"][0] * u.Gyr).value
z_end = cosmology.z_at_value(LCDM.age, res["t"][-1] * u.Gyr).value

snapshot_times = [0.16, 0.16, 0.21, 0.276,  0.8]  # Gyr — high-z main/zoom, low-z panels

# add verticle lines for snapshot times
label = ["(a)", "(a)", "(b)", "(c)", "(d)"]
for i, t in enumerate(snapshot_times):
    # Skip only true duplicate times to avoid overplotting labels/lines.
    if i > 0 and np.isclose(snapshot_times[i], snapshot_times[i - 1]):
        continue
    ax[0].axvline(x=snapshot_times[i], lw=2, alpha=0.5, color="grey")
    ax[1].axvline(x=snapshot_times[i], lw=2, alpha=0.5, color="grey")
    ax[0].text(
        snapshot_times[i],
        5.25,
        label[i],
        ha="center",
        va="center",
        fontsize=11,
    )

# keep only times within the integration window
snapshot_times = [t for t in snapshot_times if res["t"].min() < t < res["t"].max()]

# trajectory is shown in the same frame over time.
traj_Eg_all = np.asarray(res["egy_cgm"])
traj_Mh_all = np.asarray(res["m_cgm_hot"])

# Main panel uses a wide fixed domain; zoom panels use snapshot-centered square windows.
zoom_pad_dex = 0.45
main_grid_n = 70
zoom_grid_n = 160
grids = [
    (np.geomspace(5e3, 1e11, main_grid_n), np.geomspace(1e49, 2e58, main_grid_n))
]
for t_ref in snapshot_times[1:]:
    idx_ref = int(np.argmin(np.abs(np.asarray(res["t"]) - t_ref)))
    m_ref = max(float(res["m_cgm_hot"][idx_ref]), 1e-30)
    e_ref = max(float(res["egy_cgm"][idx_ref]), 1e-30)
    log_m_ref = np.log10(m_ref)
    log_e_ref = np.log10(e_ref)
    m_grid = np.geomspace(
        10 ** (log_m_ref - zoom_pad_dex),
        10 ** (log_m_ref + zoom_pad_dex),
        zoom_grid_n,
    )
    e_grid = np.geomspace(
        10 ** (log_e_ref - zoom_pad_dex),
        10 ** (log_e_ref + zoom_pad_dex),
        zoom_grid_n,
    )
    grids.append((m_grid, e_grid))

# #### loop to save for quick aesthetic updates

fp_results = []  # to store fixed-point data for each snapshot for later summary plots
axes_list = []  # to store axes for later use (e.g. inset streamplot)
dEg_list = []
dMh_list = []
for i, t_ref in enumerate(snapshot_times):
    z_label = cosmology.z_at_value(LCDM.age, t_ref * u.Gyr)
    print(f"\n── phase plane at t = {t_ref:.2f} Gyr z = {z_label:.1f} ──")
    Mh, Eg, dMh, dEg, dg = phase_plane_field(
        model,
        t_ref,
        res,
        grids[i][0],  # m_cgm_hot_grid_global
        grids[i][1],  # e_grid_global
    )

    # Classification uses the same Mh/Eg grid used to build this panel's vector field.
    fp_info = plot_phase_plane(Mh, Eg, dMh, dEg, dg, res, t_ref, plot=False, model=model)
    fp_results.append(fp_info)
    dMh_list.append(dMh)  # store for later summary plots
    dEg_list.append(dEg)

# ### END loop to save for quick aesthetic updates

# Use color scale determined by the first zoomed-in view (panel a, fp_results[1])
color_vals_first_zoom = np.ravel(fp_results[1]["color"])
finite_mask = np.isfinite(color_vals_first_zoom)
if np.any(finite_mask):
    shared_stream_norm = mcolors.Normalize(
        vmin=np.nanmin(color_vals_first_zoom[finite_mask]),
        vmax=np.nanmax(color_vals_first_zoom[finite_mask]),
    )
else:
    shared_stream_norm = None

metric_vals_all = np.concatenate([np.ravel(fp["selection_metric_log"]) for fp in fp_results])
metric_finite_mask = np.isfinite(metric_vals_all)
if np.any(metric_finite_mask):
    shared_metric_norm = mcolors.Normalize(
        vmin=np.nanmin(metric_vals_all[metric_finite_mask]),
        vmax=np.nanmax(metric_vals_all[metric_finite_mask]),
    )
else:
    shared_metric_norm = None


#### inset axes for phase plots, main map
ax_inset = ax[0].inset_axes([1.2, 0, 1.0, 1])
x_vals = fp_results[0]["x_vals"]
y_vals = fp_results[0]["y_vals"]
dlog_Mh = fp_results[0]["dlog_Mh"]
dlog_Eg = fp_results[0]["dlog_Eg"]
log_magnitude = fp_results[0]["color"]
log_Mh = fp_results[0]["log_Mh"]
log_Eg = fp_results[0]["log_Eg"]
dMh = dMh_list[0]
dEg = dEg_list[0]
strm = ax_inset.streamplot(
    x_vals,
    y_vals,
    dlog_Mh.T,
    dlog_Eg.T,
    color=log_magnitude,
    norm=shared_stream_norm,
    cmap="jet",
    linewidth=0.8,
    density=2,
    arrowsize=0.8,
)
ax_inset.plot(
    np.log10(traj_Mh_all),
    np.log10(traj_Eg_all),
    color="k",
    lw=3,
    ls="-",
    alpha=0.8,
    zorder=4,
    label=r"$z = {:.0f} - {:.1f}$".format(z_start, z_end),
)
# add a point for the current time
idx0 = int(np.argmin(np.abs(np.asarray(res["t"]) - snapshot_times[0])))
idx1 = int(np.argmin(np.abs(np.asarray(res["t"]) - snapshot_times[1])))
idx_lowz = int(np.argmin(np.abs(np.asarray(res["t"]) - snapshot_times[2])))
idx_c = int(np.argmin(np.abs(np.asarray(res["t"]) - snapshot_times[3])))
idx_d = int(np.argmin(np.abs(np.asarray(res["t"]) - snapshot_times[4])))
# add contour where they are 0
# ax_inset.contour(
#     log_Mh.T,
#     log_Eg.T,
#     dMh.T,  # this is dMh, not dlog_Mh, because we want the true nullcline where dMh=0, not where dlog_Mh=0
#     levels=[0],
#     colors="dodgerblue",
#     linewidths=2,
#     linestyles="-",
# )
# ax_inset.contour(
#     log_Mh.T,
#     log_Eg.T,
#     dEg.T,  # this is dEg, not dlog_Eg, because we want the true nullcline where dEg=0, not where dlog_Eg=0
#     levels=[0],
#     colors="tomato",
#     linewidths=2,
#     linestyles="-",
# )
# ax_inset.plot(
#     np.log10(fp_results[0]["Mh_fp"]),
#     np.log10(fp_results[0]["Eg_fp"]),
#     "o",
#     color="black",
#     markeredgecolor="k",
#     ms=5,
#     zorder=11,
# )

ax_inset.plot(
    np.log10(res["m_cgm_hot"][idx0]),
    np.log10(res["egy_cgm"][idx0]),
    ">",
    color="cyan",
    markeredgecolor="k",
    ms=8,
    zorder=10,
)
# add a unique marker for each of the snapshot times in this master big plot
ax_inset.plot(np.log10(res["m_cgm_hot"][idx_lowz]), np.log10(res["egy_cgm"][idx_lowz]), ">", color="magenta", markeredgecolor="k", ms=8, zorder=10)
ax_inset.plot(np.log10(res["m_cgm_hot"][idx_c]), np.log10(res["egy_cgm"][idx_c]), ">", color="gold", markeredgecolor="k", ms=8, zorder=10)
ax_inset.plot(np.log10(res["m_cgm_hot"][idx_d]), np.log10(res["egy_cgm"][idx_d]), ">", color="limegreen", markeredgecolor="k", ms=8, zorder=10)

# dummy artists for legend
ax_inset.plot([], [], color="dodgerblue", lw=3, label=r"$\dot{M}_{\rm CGM,hot}=0$")
ax_inset.plot([], [], color="tomato", lw=3, label=r"$\dot{E}_{\rm CGM}=0$")
ax_inset.plot([], [], "o", color="black", markeredgecolor="k", ms=5, label="classified fixed point")
ax_inset.legend(
    frameon=False, fontsize=11, loc="upper left", ncols=4, bbox_to_anchor=(-0.05, 1.23), handletextpad=0.5, columnspacing=1.0
)
ax_inset.set(
    # xlabel=r"$\log (M_{\rm CGM,hot}$ / ${\rm M_\odot})$",
    ylabel=r"$\log E_{\rm CGM}$ [erg]",
    xlim=(x_vals.min(), x_vals.max()),
    ylim=(y_vals.min(), y_vals.max()),
)
ax_inset.text(
    0.02,
    0.95,
    "(a)",
    transform=ax_inset.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
)

# this is a zoom on the high z trajectory
ax1_high = ax_inset.inset_axes([1.15, 0, 0.65, 1])


def fp_title(i):
    return f"{fp_results[i]['fp_type']}"


x_vals1 = fp_results[1]["x_vals"]
y_vals1 = fp_results[1]["y_vals"]
dlog_Mh1 = fp_results[1]["dlog_Mh"]
dlog_Eg1 = fp_results[1]["dlog_Eg"]
log_magnitude1 = fp_results[1]["color"]
log_Mh1 = fp_results[1]["log_Mh"]
log_Eg1 = fp_results[1]["log_Eg"]
strm1 = ax1_high.streamplot(
    x_vals1,
    y_vals1,
    dlog_Mh1.T,
    dlog_Eg1.T,
    color=log_magnitude1,
    norm=shared_stream_norm,
    cmap="jet",
    linewidth=0.8,
    density=1.25,
    arrowsize=0.8,
)
ax1_high.plot(
    np.log10(traj_Mh_all), np.log10(traj_Eg_all), color="k", lw=3, ls="-", alpha=0.6, zorder=4
)
ax1_high.plot(
    np.log10(res["m_cgm_hot"][idx1]),
    np.log10(res["egy_cgm"][idx1]),
    ">",
    color="cyan",
    markeredgecolor="k",
    ms=8,
    zorder=10,
)
ax1_high.contour(
    log_Mh1.T,
    log_Eg1.T,
    dMh_list[1].T,
    levels=[0],
    colors="dodgerblue",
    linewidths=3,
    linestyles="-",
)
ax1_high.contour(
    log_Mh1.T,
    log_Eg1.T,
    dEg_list[1].T,
    levels=[0],
    colors="tomato",
    linewidths=3,
    linestyles="-",
)

ax1_high.plot(
    np.log10(fp_results[1]["Mh_fp"]),
    np.log10(fp_results[1]["Eg_fp"]),
    "o",
    color="black",
    markeredgecolor="k",
    ms=5,
    zorder=11,
)
ax1_high.set(
    xlim=(x_vals1.min(), x_vals1.max()),
    ylim=(y_vals1.min(), y_vals1.max()),
)
ax1_high.set_title(fp_title(1), fontsize=11, pad=2)
ax1_high.text(
    0.02,
    0.95,
    "(a')",
    transform=ax1_high.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
)
# this is where we indicat this is an inset axis
ax_inset.indicate_inset_zoom(ax1_high, edgecolor="black", lw=2)

# 2nd z phase plane 

ax2 = ax[1].inset_axes([1.2, 0.18, 0.5, 0.75])
x_vals2 = fp_results[2]["x_vals"]
y_vals2 = fp_results[2]["y_vals"]
dlog_Mh2 = fp_results[2]["dlog_Mh"]
dlog_Eg2 = fp_results[2]["dlog_Eg"]
log_magnitude2 = fp_results[2]["color"]
log_Mh2 = fp_results[2]["log_Mh"]
log_Eg2 = fp_results[2]["log_Eg"]
strm3 = ax2.streamplot(
    x_vals2,
    y_vals2,
    dlog_Mh2.T,
    dlog_Eg2.T,
    color=log_magnitude2,
    norm=shared_stream_norm,
    cmap="jet",
    linewidth=0.8,
    density=1.25,
    arrowsize=0.8,
)
ax2.plot(
    np.log10(traj_Mh_all), np.log10(traj_Eg_all), color="k", lw=3, ls="-", alpha=0.6, zorder=4
)
ax2.plot(
    np.log10(res["m_cgm_hot"][idx_lowz]),
    np.log10(res["egy_cgm"][idx_lowz]),
    ">",
    color="magenta",
    markeredgecolor="k",
    ms=8,
    zorder=10,
)
ax2.contour(
    log_Mh2.T,
    log_Eg2.T,
    dMh_list[2].T,
    levels=[0],
    colors="dodgerblue",
    linewidths=3,
    linestyles="-",
)
ax2.contour(
    log_Mh2.T,
    log_Eg2.T,
    dEg_list[2].T,
    levels=[0],
    colors="tomato",
    linewidths=3,
    linestyles="-",
)
ax2.plot(
    np.log10(fp_results[2]["Mh_fp"]),
    np.log10(fp_results[2]["Eg_fp"]),
    "o",
    color="black",
    markeredgecolor="k",
    ms=5,
   
)
# ax2.plot([], [], color="dodgerblue", lw=1.5, label=r"$\dot{M}_{\rm CGM,hot}=0$")
# ax2.plot([], [], color="tomato", lw=1.5, label=r"$\dot{E}_{\rm CGM}=0$")
# ax2.legend(frameon=False, fontsize=10, loc="upper left", ncols=3, bbox_to_anchor=(0.0, 1.2))
ax2.set(
    ylabel=r"$\log E_{\rm CGM}$ [erg]",
    xlim=(x_vals2.min(), x_vals2.max()),
    ylim=(y_vals2.min(), y_vals2.max()),
    xlabel=r"$\log M_{\rm CGM,hot}$ [${\rm M_\odot}]$",
)
ax2.set_title(fp_title(2), fontsize=11, pad=2)
ax2.text(
    0.02,
    0.95,
    "(b)",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
)

# 3rd z phase plane
ax3 = ax[1].inset_axes([1.85, 0.18, 0.5, 0.75])
x_vals3 = fp_results[3]["x_vals"]
y_vals3 = fp_results[3]["y_vals"]
dlog_Mh3 = fp_results[3]["dlog_Mh"]
dlog_Eg3 = fp_results[3]["dlog_Eg"]
log_magnitude3 = fp_results[3]["color"]
log_Mh3 = fp_results[3]["log_Mh"]
log_Eg3 = fp_results[3]["log_Eg"]
strm4 = ax3.streamplot(
    x_vals3,
    y_vals3,
    dlog_Mh3.T,
    dlog_Eg3.T,
    color=log_magnitude3,
    norm=shared_stream_norm,
    cmap="jet",
    linewidth=0.8,
    density=1.25,
    arrowsize=0.8,
)
ax3.plot(
    np.log10(traj_Mh_all), np.log10(traj_Eg_all), color="k", lw=3, ls="-", alpha=0.6, zorder=4
)
ax3.plot(
    np.log10(res["m_cgm_hot"][idx_c]),
    np.log10(res["egy_cgm"][idx_c]),
    ">",
    color="gold",
    markeredgecolor="k",
    ms=8,
    zorder=10,
)
ax3.contour(
    log_Mh3.T,
    log_Eg3.T,
    dMh_list[3].T,
    levels=[0],
    colors="dodgerblue",
    linewidths=3,
    linestyles="-",
)
ax3.contour(
    log_Mh3.T,
    log_Eg3.T,
    dEg_list[3].T,
    levels=[0],
    colors="tomato",
    linewidths=3,
    linestyles="-",
)
ax3.plot(
    np.log10(fp_results[3]["Mh_fp"]),
    np.log10(fp_results[3]["Eg_fp"]),
    "o",
    color="black",
    markeredgecolor="k",
    ms=5,
    zorder=11,
)
ax3.set(
    xlim=(x_vals3.min(), x_vals3.max()),
    ylim=(y_vals3.min(), y_vals3.max()),
    xlabel=r"$\log M_{\rm CGM,hot}$ [${\rm M_\odot}]$",
)
ax3.set_title(fp_title(3), fontsize=11, pad=2)
ax3.text(
    0.02,
    0.95,
    "(c)",
    transform=ax3.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
)

# 4th z phase plane
ax4 = ax[1].inset_axes([2.5, 0.18, 0.5, 0.75])
x_vals4 = fp_results[4]["x_vals"]
y_vals4 = fp_results[4]["y_vals"]
dlog_Mh4 = fp_results[4]["dlog_Mh"]
dlog_Eg4 = fp_results[4]["dlog_Eg"]
log_magnitude4 = fp_results[4]["color"]
log_Mh4 = fp_results[4]["log_Mh"]
log_Eg4 = fp_results[4]["log_Eg"]
strm5 = ax4.streamplot(
    x_vals4,
    y_vals4,
    dlog_Mh4.T,
    dlog_Eg4.T,
    color=log_magnitude4,
    norm=shared_stream_norm,
    cmap="jet",
    linewidth=0.8,
    density=1.5,
    arrowsize=0.8,
)
ax4.plot(
    np.log10(traj_Mh_all), np.log10(traj_Eg_all), color="k", lw=3, ls="-", alpha=0.6, zorder=4
)
ax4.plot(
    np.log10(res["m_cgm_hot"][idx_d]),
    np.log10(res["egy_cgm"][idx_d]),
    ">",
    color="limegreen",
    markeredgecolor="k",
    ms=8,
    zorder=10,
)
ax4.contour(
    log_Mh4.T,
    log_Eg4.T,
    dMh_list[4].T,
    levels=[0],
    colors="dodgerblue",
    linewidths=3,
    linestyles="-",
)
ax4.contour(
    log_Mh4.T,
    log_Eg4.T,
    dEg_list[4].T,
    levels=[0],
    colors="tomato",
    linewidths=3,
    linestyles="-",
)
ax4.plot(
    np.log10(fp_results[4]["Mh_fp"]),
    np.log10(fp_results[4]["Eg_fp"]),
    "o",
    color="black",
    markeredgecolor="k",
    ms=5,
    zorder=11,
)
ax4.set(
    xlim=(x_vals4.min(), x_vals4.max()),
    ylim=(y_vals4.min(), y_vals4.max()),
    xlabel=r"$\log M_{\rm CGM,hot}$ [${\rm M_\odot}]$",
)
ax4.set_title(fp_title(4), fontsize=11, pad=2)
ax4.text(
    0.02,
    0.95,
    "(d)",
    transform=ax4.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
)
cbar_ax = ax1_high.inset_axes([1.08, -1.1, 0.08, 2.1])
cbar = fig.colorbar(
    strm.lines,
    cax=cbar_ax,
    ax=[ax_inset, ax1_high, ax2, ax3, ax4],
    pad=0.02,
    fraction=0.03,
)
cbar.set_label(
    r"$\log\sqrt{(d\log M_{\rm CGM,hot}/dt)^2 + (d\log E_{\rm CGM}/dt)^2}$ [Gyr$^{-1}$]"
)
# custom legend 
down_and_left = r"$\swarrow$: cooling halo, mass depletion into ISM"
down_and_right = r"$\searrow$: accretion dominated, denser and colder CGM"
up_and_left = r"$\nwarrow$: pressurized CGM, halo outflows"
up_and_right = r"$~~~~~\nearrow$: hot halo, IGM accretion and SNe feedback"

# add as text
ax3.text(
    0.5,
    -0.43,
    f"{down_and_left}\t{down_and_right}\n\n {up_and_left}\t{up_and_right}",
    transform=ax3.transAxes,
    ha="center",
    va="center",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=5),
)
plt.savefig("final_figs/appendix_phase_plane_1e12.pdf", dpi=200, bbox_inches="tight")
plt.show()



# %% f_prevent phase space analysis


# Select specific f_prevent floors to display and their corresponding times
selected_phase_floors = [0.01, 0.2,  0.5, 0.6, 1.0]
create_at_time = [0.3, 0.3, 0.325, 0.4, 0.24]  # Customize these times [Gyr]
phase_xlim = (6.25, 7.74)
phase_ylim = (53.5, 55.5)
t_span = (0.10, 2.50)  # Gyr
f_prevent_floor_to_try = [1e-6, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 1.0]
colors = plt.cm.Dark2_r(np.linspace(0, 1, len(f_prevent_floor_to_try))) # to mathc appendix_plotting colors

mhalo_0 = 1e12 * u.Msun
##++++++++++++++++++++++++++++++++++++ 
### keep all per-floor products in memory for later reuse.
results_list = []
f_prevent_phase_memory = []

# Run models for all f_prevent values
for idx, f_prevent_floor in enumerate(f_prevent_floor_to_try):
    model_ = CGMRegulator(
        mhalo_0,
        t_span,
        tstep=0.001,
        add_f_prevent_floor=f_prevent_floor,
        verbose=False,
        updated_loadings=True,
        updated_halo_infall=True,
    )
    model_.run_halo()

    results_ = model_.get_results()
    derived_ = model_.get_derived_quantities()

    # Store results with color for trajectory plotting
    results_list.append(
        {
            "idx": idx,
            "f_prevent_floor": f_prevent_floor,
            "label": f"{f_prevent_floor:g}",
            "color": colors[idx],
            "results": results_,
            "derived": derived_,
            "model": model_,  # Store model for later phase plane computation
        }
    )
##Build phase portraits at specified times for selected floors
selected_phase_memory = []
for target_floor, t_phase_ref in zip(selected_phase_floors, create_at_time):
    # Find the matching result from the full run
    match = next(
        (
            item
            for item in results_list
            if np.isclose(item["f_prevent_floor"], target_floor)
        ),
        None,
    )
    
    if match is None:
        warnings.warn(f"Requested f_prevent floor {target_floor} not found")
        continue
    
    # Get the model and results
    model_ = match["model"]
    results_ = match["results"]
    
    # Find the closest time index
    t_arr = np.asarray(results_["t"])
    idx_phase = int(np.argmin(np.abs(t_arr - t_phase_ref)))
    t_phase_actual = float(t_arr[idx_phase])
    
    # Build phase portrait at this specific time
    phase_grid_n = 65
    m_grid = np.geomspace(
        10**phase_xlim[0],
        10**phase_xlim[1],
        phase_grid_n,
    )
    e_grid = np.geomspace(
        10**phase_ylim[0],
        10**phase_ylim[1],
        phase_grid_n,
    )
    
    Mh, Eg, dMh, dEg, dg = phase_plane_field(
        model_,
        t_phase_actual,
        results_,
        m_grid,
        e_grid,
    )
    fp_info = plot_phase_plane(
        Mh,
        Eg,
        dMh,
        dEg,
        dg,
        results_,
        t_phase_actual,
        plot=False,
        show_trajectory=False,
        model=model_,
    )

    selected_phase_memory.append(
        {
            "f_prevent_floor": target_floor,
            "t_phase_requested": t_phase_ref,
            "t_phase_actual": t_phase_actual,
            "phase_idx": idx_phase,
            "Mh": Mh,
            "Eg": Eg,
            "dMh": dMh,
            "dEg": dEg,
            "derived_grid": dg,
            "fp_info": fp_info,
            "results": results_,
            "color": match["color"],
        }
    )
###++++++++++++++++++++++++++++++++++++ 

# create single row figure with 5 phase portraits
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 3), dpi=300, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.0, wspace=0.02)

for ax in axes:
    ax.grid(which="both", ls="-", lw=0.5, alpha=0.35, zorder=0)

# Compute shared color normalization across all phase portraits
all_phase_colors = np.concatenate(
    [np.ravel(item["fp_info"]["color"]) for item in selected_phase_memory]
) if len(selected_phase_memory) > 0 else np.array([])
finite_mask = np.isfinite(all_phase_colors)
if np.any(finite_mask):
    phase_norm = mcolors.Normalize(
        vmin=np.nanmin(all_phase_colors[finite_mask]),
        vmax=np.nanmax(all_phase_colors[finite_mask]),
    )
else:
    phase_norm = None

phase_panel_labels = ["(j)", "(k)", "(l)", "(m)", "(n)"]

strm_last = None

for i_phase, item in enumerate(selected_phase_memory):
    axp = axes[i_phase]
    fp_info = item["fp_info"]
    fp_type = fp_info["fp_type"]
    dMh = item["dMh"]
    dEg = item["dEg"]
    res_floor = item["results"]
    idx_phase = item["phase_idx"]
    traj_color = plt.cm.Dark2_r(np.linspace(0, 1,5))[i_phase]
    t_phase_actual = item["t_phase_actual"]

    # Plot streamlines
    strm_last = axp.streamplot(
        fp_info["x_vals"],
        fp_info["y_vals"],
        fp_info["dlog_Mh"].T,
        fp_info["dlog_Eg"].T,
        color=fp_info["color"],
        cmap="jet",
        norm=phase_norm,
        linewidth=0.7,
        density=1.2,
        arrowsize=0.7,
    )

    # Plot trajectory in matching color
    traj_E = np.asarray(res_floor["egy_cgm"])
    traj_M = np.asarray(res_floor["m_cgm_hot"])
    pos_mask = (traj_E > 0) & (traj_M > 0)
    axp.plot(
        np.log10(traj_M[pos_mask]),
        np.log10(traj_E[pos_mask]),
        color=traj_color,
        lw=2.0,
        alpha=0.8,
        zorder=5,
    )

    # Plot nullclines
    axp.contour(
        fp_info["log_Mh"].T,
        fp_info["log_Eg"].T,
        dMh.T,
        levels=[0],
        colors="dodgerblue",
        linewidths=1.4,
    )
    axp.contour(
        fp_info["log_Mh"].T,
        fp_info["log_Eg"].T,
        dEg.T,
        levels=[0],
        colors="tomato",
        linewidths=1.4,
    )

    # Plot current position at the specified time
    ref_x = np.log10(res_floor["m_cgm_hot"][idx_phase])
    ref_y = np.log10(res_floor["egy_cgm"][idx_phase])
    axp.plot(ref_x, ref_y, "o", color="white", markeredgecolor="k", ms=6, zorder=10)
    
    # Plot fixed point
    axp.plot(
        np.log10(fp_info["Mh_fp"]),
        np.log10(fp_info["Eg_fp"]),
        "o",
        color="black",
        markeredgecolor="k",
        ms=5,
        zorder=11,
    )

    axp.set_xlim(*phase_xlim)
    axp.set_ylim(*phase_ylim)
    
    # Calculate redshift at this time
    z_phase = cosmology.z_at_value(LCDM.age, t_phase_actual * u.Gyr)
    
    axp.set_title(
        rf"$f_{{\rm prevent,floor}}={item['f_prevent_floor']:g}$",
        fontsize=11,
        pad=5,
    )
    
    # Add fixed point classification
    axp.text(
        0.98,
        0.04,
        fp_type,
        transform=axp.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    
    # Add panel label
    axp.text(
        0.02,
        0.02,
        phase_panel_labels[i_phase],
        transform=axp.transAxes,
        fontsize=12,
        ha="left",
        va="bottom",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )
    
    # Add redshift annotation
    axp.text(
        0.1,
        0.95,
        rf"$z={z_phase.value:.2f}$" "\n"rf"$t_{{\rm univ}}={t_phase_actual:.2f}$ Gyr",
        transform=axp.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=1.5),
    )
    
    # Set labels
    if i_phase == 0:
        axp.set_ylabel(r"$\log E_{\rm CGM}$ [erg]")


axes[2].set_xlabel(r"$\log M_{\rm CGM,hot}$ [${\rm M_\odot}$]")

# Add colorbar below the entire figure

cbar_ax = axes[-1].inset_axes([1.04, 0, 0.05, 1], transform=axes[-1].transAxes)
cbar = fig.colorbar(strm_last.lines, cax=cbar_ax)
cbar.set_label(
    r"$\log\sqrt{(d\log M_{\rm CGM,hot}/dt)^2 + (d\log E_{\rm CGM}/dt)^2}$ [Gyr$^{-1}$]",
    fontsize=9,
)

# Add nullcline legend to first panel
axes[0].plot([], [], color="dodgerblue", lw=1.5, label=r"$\dot{M}_{\rm CGM,hot}=0$")
axes[0].plot([], [], color="tomato", lw=1.5, label=r"$\dot{E}_{\rm CGM}=0$")
axes[0].plot([], [], "o", color="white", markeredgecolor="k", ms=6, label="point in evolution")
axes[0].plot([], [], "o", color="black", markeredgecolor="k", ms=5, label="classified fixed point")

axes[0].legend(frameon=False, fontsize=11, loc="lower left", ncol=4, bbox_to_anchor=(0.6, 1.05))

plt.savefig("final_figs/phase_portraits_fprev_comparison_Mhalo{:.2e}.pdf".format(mhalo_0.value), dpi=300, bbox_inches="tight")
plt.show()
# plt.savefig("final_figs/f_prevent_rates_plus_phase_t0p4_2x5.pdf", dpi=300, bbox_inches="tight")
# plt.show()

# %%

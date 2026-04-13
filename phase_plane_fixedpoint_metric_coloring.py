import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy import cosmology
import astropy.units as u

from cgm_sf_regulator import CGMRegulator


H0 = 70
LCDM = cosmology.LambdaCDM(H0=H0, Om0=0.3, Ode0=0.7)


plt.rcParams.update(
    {
        "text.usetex": True,
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


def phase_plane_field(model, t_ref, res, m_cgm_hot_grid, e_cgm_grid):
    t_arr = np.asarray(res["t"])
    idx = int(np.argmin(np.abs(t_arr - t_ref)))

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

    ny, nx = Mh.shape
    for i in range(ny):
        for j in range(nx):
            state = np.array(
                [
                    m_ism_0,
                    m_star_0,
                    m_bulge_0,
                    float(Mh[i, j]),
                    m_cgm_cold_0,
                    m_metals_cgm_0,
                    m_metals_ism_0,
                    m_halo_0,
                    float(Eg[i, j]),
                ]
            )
            derivs = model.mass_evolution(float(t_ref), state, ode_mode=True)
            dMh[i, j] = derivs[3]
            dEg[i, j] = derivs[8]

    return Mh, Eg, dMh, dEg


def fixed_point_selection_metric(Mh, Eg, dMh, dEg):
    """Build the metric actually used in fixed-point selection.

    If crossing cells exist, metric is the crossing-cell score and only crossing cells are finite.
    If no crossings exist, metric is the point-wise flow norm (converted to cell values by corner average).
    """
    dMh_00, dMh_10 = dMh[:-1, :-1], dMh[1:, :-1]
    dMh_01, dMh_11 = dMh[:-1, 1:], dMh[1:, 1:]
    dEg_00, dEg_10 = dEg[:-1, :-1], dEg[1:, :-1]
    dEg_01, dEg_11 = dEg[:-1, 1:], dEg[1:, 1:]

    dMh_min = np.minimum.reduce([dMh_00, dMh_10, dMh_01, dMh_11])
    dMh_max = np.maximum.reduce([dMh_00, dMh_10, dMh_01, dMh_11])
    dEg_min = np.minimum.reduce([dEg_00, dEg_10, dEg_01, dEg_11])
    dEg_max = np.maximum.reduce([dEg_00, dEg_10, dEg_01, dEg_11])

    cross_Mh = np.isfinite(dMh_min) & np.isfinite(dMh_max) & (dMh_min <= 0.0) & (dMh_max >= 0.0)
    cross_Eg = np.isfinite(dEg_min) & np.isfinite(dEg_max) & (dEg_min <= 0.0) & (dEg_max >= 0.0)
    both = cross_Mh & cross_Eg

    cell_score = 0.25 * (
        np.hypot(dMh_00, dEg_00)
        + np.hypot(dMh_10, dEg_10)
        + np.hypot(dMh_01, dEg_01)
        + np.hypot(dMh_11, dEg_11)
    )

    if np.any(both):
        metric = np.where(both, cell_score, np.nan)
        reason = "crossing-cell score (finite only where both nullclines change sign)"
        i, j = np.unravel_index(np.nanargmin(metric), metric.shape)
        Mh_fp = np.sqrt(Mh[i, j] * Mh[i + 1, j + 1])
        Eg_fp = np.sqrt(Eg[i, j] * Eg[i + 1, j + 1])
    else:
        # Convert point metric to cell metric by averaging 4 corners so pcolormesh can render cells.
        flow_norm = np.hypot(dMh, dEg)
        metric = 0.25 * (
            flow_norm[:-1, :-1]
            + flow_norm[1:, :-1]
            + flow_norm[:-1, 1:]
            + flow_norm[1:, 1:]
        )
        reason = "minimum flow magnitude (no crossing cells found)"
        i, j = np.unravel_index(np.nanargmin(flow_norm), flow_norm.shape)
        Mh_fp = Mh[i, j]
        Eg_fp = Eg[i, j]

    return metric, reason, Mh_fp, Eg_fp


def classify_point(model, res, t_ref, Mh_fp, Eg_fp):
    """Classify local fixed-point type from Jacobian eigenvalues at a chosen point."""
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

    deltaM = max(abs(Mh_fp) * 1e-5, 1e2)
    deltaE = max(abs(Eg_fp) * 1e-5, 1e36)

    def eval_2d(dm, de):
        s = state_fp.copy()
        s[3] += dm
        s[8] += de
        d = model.mass_evolution(float(t_ref), s, ode_mode=True)
        return d[3], d[8]

    J = np.zeros((2, 2))
    fp, fm = eval_2d(deltaM, 0), eval_2d(-deltaM, 0)
    J[0, 0] = (fp[0] - fm[0]) / (2 * deltaM)
    J[1, 0] = (fp[1] - fm[1]) / (2 * deltaM)
    fp, fm = eval_2d(0, deltaE), eval_2d(0, -deltaE)
    J[0, 1] = (fp[0] - fm[0]) / (2 * deltaE)
    J[1, 1] = (fp[1] - fm[1]) / (2 * deltaE)

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

    return fp_type, eigvals, tr, det, disc, J


def main():
    mhalo_z0 = 1e12 * u.Msun
    t_span = (0.1, 13.3)

    model = CGMRegulator(
        mhalo_z0,
        t_span,
        tstep=0.005,
        add_f_prevent_floor=1e-8,
        KS_kappa_s=0.02,
        KS_n=1.8,
        disk_scale_length=0.018,
        KS_parametrization="KS1998",
        TEST_tej_Tvir_definition=False,
        eta_z=0.7,
    )
    model.run_halo()
    res = model.get_results()

    snapshot_times = [0.16, 0.16, 0.21, 0.276, 0.8]
    t_ref = snapshot_times[1]  # a' panel

    idx_ref = int(np.argmin(np.abs(np.asarray(res["t"]) - t_ref)))
    m_ref = max(float(res["m_cgm_hot"][idx_ref]), 1e-30)
    e_ref = max(float(res["egy_cgm"][idx_ref]), 1e-30)

    zoom_pad_dex = 0.45
    
    zoom_grid_n = 160
    m_grid = np.geomspace(10 ** (np.log10(m_ref) - zoom_pad_dex), 10 ** (np.log10(m_ref) + zoom_pad_dex), zoom_grid_n)
    e_grid = np.geomspace(10 ** (np.log10(e_ref) - zoom_pad_dex), 10 ** (np.log10(e_ref) + zoom_pad_dex), zoom_grid_n)

    Mh, Eg, dMh, dEg = phase_plane_field(model, t_ref, res, m_grid, e_grid)
    metric, reason, Mh_fp, Eg_fp = fixed_point_selection_metric(Mh, Eg, dMh, dEg)
    fp_type, eigvals, tr, det, disc, J = classify_point(model, res, t_ref, Mh_fp, Eg_fp)

    log_Mh = np.log10(Mh)
    log_Eg = np.log10(Eg)
    x_edges = np.log10(m_grid)
    y_edges = np.log10(e_grid)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=220)

    finite = np.isfinite(metric)
    metric_to_plot = np.log10(metric + 1e-60)
    if np.any(finite):
        vmin = np.nanmin(metric_to_plot[finite])
        vmax = np.nanmax(metric_to_plot[finite])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        metric_to_plot.T,
        cmap="viridis",
        norm=norm,
        shading="auto",
    )

    ax.contour(log_Mh.T, log_Eg.T, dMh.T, levels=[0], colors="dodgerblue", linewidths=1.6)
    
    ax.contour(log_Mh.T, log_Eg.T, dEg.T, levels=[0], colors="tomato", linewidths=1.6)

    # Overlay uncolored streamlines in log-phase coordinates.
    ln10 = np.log(10.0)
    x_vals = log_Mh[:, 0]
    y_vals = log_Eg[0, :]
    dlog_Mh = dMh / (Mh * ln10)
    dlog_Eg = dEg / (Eg * ln10)
    strm = ax.streamplot(
        x_vals,
        y_vals,
        dlog_Mh.T,
        dlog_Eg.T,
        color="k",
        linewidth=0.55,
        density=1.2,
        arrowsize=0.7,
        zorder=4,
    )

    ax.plot(np.log10(Mh_fp), np.log10(Eg_fp), "o", color="black", markeredgecolor="white", ms=6, zorder=5)

    z_ref = cosmology.z_at_value(LCDM.age, t_ref * u.Gyr)
    ax.set_title(rf"a' metric map at $t={t_ref:.2f}$ Gyr ($z={z_ref:.2f}$)")
    ax.set_xlabel(r"$\log M_{\rm CGM,hot}$ [${\rm M_\odot}$]")
    ax.set_ylabel(r"$\log E_{\rm CGM}$ [erg]")

    cax_metric = ax.inset_axes([1.02, 0.54, 0.04, 0.44])
    cbar_metric = fig.colorbar(pcm, cax=cax_metric)
    cbar_metric.set_label(
        r"$\log_{10}(\mathrm{selection\ metric})$: "
        r"crossing score $=\langle\sqrt{\dot{M}^2+\dot{E}^2}\rangle_{\rm cell}$; "
        r"fallback $=\sqrt{\dot{M}^2+\dot{E}^2}$"
    )

    print("[metric-map] reason:", reason)
    print(
        "[metric-map] selected point:",
        f"log10(Mh)={np.log10(Mh_fp):.4f}, log10(Eg)={np.log10(Eg_fp):.4f}",
    )
    print(
        "[metric-map] classification:",
        fp_type,
        f"| eigvals={eigvals}",
        f"| tr={tr:.6e}",
        f"| det={det:.6e}",
        f"| disc={disc:.6e}",
    )
    print("[metric-map] Jacobian:\n", J)

    fig.tight_layout()
    plt.savefig("final_figs/a_prime_fixedpoint_metric_map.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

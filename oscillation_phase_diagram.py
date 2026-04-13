# %% Oscillation phase-diagram analysis
# Runs a grid of CGMRegulator models varying halo mass, loading factors,
# and SF efficiency.  For each run it measures oscillation strength
# (sigma of detrended log-SFR residuals, dominant FFT period) and records
# time-averaged physical diagnostics (tcool/tdyn, CGM temperature, etc.).
# The output is a structured table + four summary plots that reveal *where*
# in physical-parameter space the solution oscillates.

import importlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy import cosmology
import astropy.units as u
import astropy.constants as consts
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.signal import welch

import cgm_sf_regulator
from cgm_sf_regulator import CGMRegulator, mhalo_at_z0_fakhouri

importlib.reload(cgm_sf_regulator)

plt.rcParams.update(
    {
        "text.usetex": False,
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
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

# ── cosmology ────────────────────────────────────────────────────────
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


# ── helpers ──────────────────────────────────────────────────────────
def exp_fit(t, a, tau, c):
    """Exponential trend for detrending log-SFR."""
    return a * np.exp(t / tau) + c


def measure_oscillation(t, sfr_linear, sfr_bin_width=0.01, t_burnin_frac=0.2):
    """Detrend log-SFR with exp fit, return oscillation metrics.

    Parameters
    ----------
    t : array  – time in Gyr (from ODE results)
    sfr_linear : array  – SFR in Msun/Gyr (linear, **not** logged)
    sfr_bin_width : float – bin width in Gyr for binning before fit
    t_burnin_frac : float – fraction of time span to skip as transient

    Returns
    -------
    dict with keys:
        sigma_osc       – std of log(SFR/fit)
        amplitude_osc   – peak-to-peak / mean of detrended SFR
        dominant_period  – dominant oscillation period [Gyr] from Welch PSD
        fit_success      – bool, whether the exp fit converged
    """
    log_sfr = np.log10(np.clip(sfr_linear, 1e-30, None))

    # burn-in mask
    t_cut = t.min() + t_burnin_frac * (t.max() - t.min())
    mask = np.isfinite(log_sfr) & (t > t_cut)
    if mask.sum() < 20:
        return dict(
            sigma_osc=np.nan,
            amplitude_osc=np.nan,
            dominant_period=np.nan,
            fit_success=False,
        )

    t_m = t[mask]
    log_sfr_m = log_sfr[mask]

    # bin
    bins = np.arange(t_m.min(), t_m.max() + sfr_bin_width, sfr_bin_width)
    sfr_binned, edges, _ = binned_statistic(t_m, log_sfr_m, statistic="mean", bins=bins)
    bc = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(sfr_binned)
    bc, sfr_binned = bc[ok], sfr_binned[ok]

    if len(bc) < 10:
        return dict(
            sigma_osc=np.nan,
            amplitude_osc=np.nan,
            dominant_period=np.nan,
            fit_success=False,
        )

    # exponential fit to binned log-SFR
    try:
        p0 = [1.0, -0.5, np.median(sfr_binned)]
        popt, _ = curve_fit(exp_fit, bc, sfr_binned, p0=p0, maxfev=10000)
        trend = exp_fit(t_m, *popt)
        fit_ok = True
    except RuntimeError:
        trend = np.median(log_sfr_m) * np.ones_like(t_m)
        fit_ok = False

    residual = log_sfr_m - trend
    sigma_osc = float(np.std(residual))

    # peak-to-peak amplitude
    sfr_detrended = 10**log_sfr_m / 10**trend
    amplitude_osc = float((sfr_detrended.max() - sfr_detrended.min()) / np.mean(sfr_detrended))

    # dominant period via Welch PSD on uniformly-resampled residual
    dt_median = float(np.median(np.diff(t_m)))
    if dt_median > 0 and len(residual) > 30:
        nperseg = min(len(residual), max(32, len(residual) // 4))
        freqs, psd = welch(residual, fs=1.0 / dt_median, nperseg=nperseg)
        if len(freqs) > 1 and np.any(psd[1:] > 0):
            peak_idx = 1 + np.argmax(psd[1:])
            dominant_period = float(1.0 / freqs[peak_idx]) if freqs[peak_idx] > 0 else np.nan
        else:
            dominant_period = np.nan
    else:
        dominant_period = np.nan

    return dict(
        sigma_osc=sigma_osc,
        amplitude_osc=amplitude_osc,
        dominant_period=dominant_period,
        fit_success=fit_ok,
    )


def extract_diagnostics(results, derived, t_burnin_frac=0.2):
    """Time-averaged physical diagnostics from a single run (post-transient)."""
    t = results["t"]
    t_cut = t.min() + t_burnin_frac * (t.max() - t.min())
    m = t > t_cut

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        tcool = derived["tcool_real"][m]
        tdyn = derived["t_dynamical"][m]
        tcool_over_tdyn = np.nanmean(tcool / tdyn)

        mean_Tcgm = np.nanmean(derived["cgm_temp"][m])
        mean_Tvir = np.nanmean(derived["halo_vir_temp"][m])
        mean_fprevent = np.nanmean(derived["f_prevent"][m])

        hot = results["m_cgm_hot"][m]
        cold = results["m_cgm_cold"][m]
        hot_cold_ratio = np.nanmean(hot / np.clip(cold, 1e-30, None))

        dot_e_wind = derived["dot_e_ism_wind"][m]
        dot_e_cool = derived["dot_e_cgm_cooling"][m]
        heating_cooling = np.nanmean(dot_e_wind / np.clip(dot_e_cool, 1e-30, None))

        mean_tcool = np.nanmean(tcool)
        mean_tdyn = np.nanmean(tdyn)
        mean_tej = np.nanmean(derived["t_ejection"][m])

        mean_mcgm = np.nanmean(results["m_cgm"][m])
        mean_mism = np.nanmean(results["m_ism"][m])
        mean_mstar = np.nanmean(results["m_star"][m])

    return dict(
        tcool_over_tdyn=float(tcool_over_tdyn),
        mean_Tcgm_K=float(mean_Tcgm),
        mean_Tvir_K=float(mean_Tvir),
        mean_fprevent=float(mean_fprevent),
        hot_cold_ratio=float(hot_cold_ratio),
        heating_cooling_ratio=float(heating_cooling),
        mean_tcool_gyr=float(mean_tcool),
        mean_tdyn_gyr=float(mean_tdyn),
        mean_tej_gyr=float(mean_tej),
        mean_mcgm=float(mean_mcgm),
        mean_mism=float(mean_mism),
        mean_mstar=float(mean_mstar),
    )


# ── grid configuration ───────────────────────────────────────────────
# Edit these arrays to control the scan.  Start small (fewer points) to
# get a quick look, then increase resolution for the paper figure.
# %%
z_obs = 6.0
t_init = 0.1  # Gyr
t_final = LCDM.age(z_obs).value  # Gyr
t_span = (t_init, t_final)

ode_solver_step = 0.0005  # Gyr  (output cadence; Radau adapts internally)
sfr_bin_width = 0.01  # Gyr
t_burnin_frac = 0.2  # skip first 20 % as transient

# --- axes to scan -------------------------------------------------
halo_masses_zobs = np.geomspace(1e9, 1e13, 12) * u.Msun
alpha_M_values = np.geomspace(0.01, 10, 10)
alpha_E_values = [0.1]           # hold fixed (or scan independently)
kappa_s_values = [0.02]          # hold fixed (or scan independently)
f_prevent_floor_values = [1e-6]  # hold fixed

# ── run the grid ─────────────────────────────────────────────────────
# %%
records = []
n_total = (
    len(halo_masses_zobs)
    * len(alpha_M_values)
    * len(alpha_E_values)
    * len(kappa_s_values)
    * len(f_prevent_floor_values)
)
counter = 0

for mhalo_zobs in halo_masses_zobs:
    for aM in alpha_M_values:
        for aE in alpha_E_values:
            for ks in kappa_s_values:
                for fp in f_prevent_floor_values:
                    counter += 1
                    mhalo_z0 = mhalo_at_z0_fakhouri(mhalo_zobs, z_obs) * u.Msun
                    print(
                        f"[{counter}/{n_total}] Mh(z={z_obs})={mhalo_zobs.value:.1e}  "
                        f"aM={aM:.2e}  aE={aE:.2e}  ks={ks:.3f}  fp={fp:.1e}"
                    )

                    try:
                        model = CGMRegulator(
                            mhalo_z0,
                            t_span,
                            tstep=ode_solver_step,
                            KS_kappa_s=ks,
                            add_f_prevent_floor=fp,
                            verbose=False,
                            alpha_m=aM,
                            alpha_e=aE,
                        )
                        model.run_halo()
                        res = model.get_results()
                        der = model.get_derived_quantities()

                        osc = measure_oscillation(
                            res["t"],
                            der["dot_m_sfr"],
                            sfr_bin_width=sfr_bin_width,
                            t_burnin_frac=t_burnin_frac,
                        )
                        diag = extract_diagnostics(
                            res, der, t_burnin_frac=t_burnin_frac
                        )
                    except Exception as exc:
                        print(f"   *** FAILED: {exc}")
                        osc = dict(
                            sigma_osc=np.nan,
                            amplitude_osc=np.nan,
                            dominant_period=np.nan,
                            fit_success=False,
                        )
                        diag = {k: np.nan for k in [
                            "tcool_over_tdyn", "mean_Tcgm_K", "mean_Tvir_K",
                            "mean_fprevent", "hot_cold_ratio",
                            "heating_cooling_ratio", "mean_tcool_gyr",
                            "mean_tdyn_gyr", "mean_tej_gyr", "mean_mcgm",
                            "mean_mism", "mean_mstar",
                        ]}

                    rec = dict(
                        mhalo_zobs=mhalo_zobs.value,
                        mhalo_z0=mhalo_z0.value,
                        alpha_M=aM,
                        alpha_E=aE,
                        kappa_s=ks,
                        f_prevent_floor=fp,
                    )
                    rec.update(osc)
                    rec.update(diag)
                    records.append(rec)
                    plt.close("all")

print(f"\nGrid complete: {len(records)} models")

# ── save table ───────────────────────────────────────────────────────
# %%
col_order = [
    "mhalo_zobs", "mhalo_z0", "alpha_M", "alpha_E", "kappa_s",
    "f_prevent_floor", "sigma_osc", "amplitude_osc", "dominant_period",
    "fit_success", "tcool_over_tdyn", "mean_Tcgm_K", "mean_Tvir_K",
    "mean_fprevent", "hot_cold_ratio", "heating_cooling_ratio",
    "mean_tcool_gyr", "mean_tdyn_gyr", "mean_tej_gyr",
    "mean_mcgm", "mean_mism", "mean_mstar",
]

header = "  ".join(f"{c:>22s}" for c in col_order)
outpath = "runs/oscillation_phase_diagram.txt"
with open(outpath, "w") as f:
    f.write("# " + header + "\n")
    for r in records:
        vals = "  ".join(f"{r.get(c, np.nan):>22.6e}" if not isinstance(r.get(c), bool)
                         else f"{int(r.get(c)):>22d}" for c in col_order)
        f.write(vals + "\n")
print(f"Saved {outpath}")


# ── helper: build arrays from records for plotting ───────────────────
def rec_array(key):
    return np.array([r[key] for r in records], dtype=float)


# ──────────────────────────────────────────────────────────────────────
# PLOT 1: 2-D phase diagram  Mhalo(z_obs) vs alpha_M, coloured by sigma_osc
# ──────────────────────────────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
sc = ax.scatter(
    rec_array("mhalo_zobs"),
    rec_array("alpha_M"),
    c=rec_array("sigma_osc"),
    cmap="inferno",
    edgecolors="k",
    linewidths=0.4,
    s=60,
)
cbar = plt.colorbar(sc, ax=ax, label=r"$\sigma_{\rm osc}$ (detrended log SFR)")
ax.set(
    xlabel=r"$M_{\rm halo}(z_{\rm obs})$ $[{\rm M_\odot}]$",
    ylabel=r"$\alpha_M$",
    xscale="log",
    yscale="log",
    title=f"Oscillation strength — $z_{{\\rm obs}}={z_obs}$",
)
plt.tight_layout()
plt.savefig("figures/oscillation_phase_Mh_alphaM.png", dpi=200, bbox_inches="tight")
plt.show()


# ──────────────────────────────────────────────────────────────────────
# PLOT 2: sigma_osc  vs  <t_cool / t_dyn>
# Tests whether the multi-dim parameter dependence collapses onto one curve
# ──────────────────────────────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
sc = ax.scatter(
    rec_array("tcool_over_tdyn"),
    rec_array("sigma_osc"),
    c=np.log10(rec_array("mhalo_zobs")),
    cmap="viridis",
    edgecolors="k",
    linewidths=0.4,
    s=50,
)
cbar = plt.colorbar(sc, ax=ax, label=r"$\log_{10} M_{\rm halo}(z_{\rm obs})$")
ax.set(
    xlabel=r"$\langle t_{\rm cool} / t_{\rm dyn} \rangle$",
    ylabel=r"$\sigma_{\rm osc}$",
    xscale="log",
    title="Instability criterion",
)
ax.axvline(1.0, ls="--", color="grey", lw=1, label=r"$t_{\rm cool}/t_{\rm dyn}=1$")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("figures/oscillation_sigma_vs_tcool_tdyn.png", dpi=200, bbox_inches="tight")
plt.show()


# ──────────────────────────────────────────────────────────────────────
# PLOT 3: dominant oscillation period vs  <t_cool>  and <t_dyn>
# Physical oscillations should scale with a physical timescale
# ──────────────────────────────────────────────────────────────────────
# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200, sharey=True)

for ax_i, (xkey, xlabel) in enumerate(
    [
        ("mean_tcool_gyr", r"$\langle t_{\rm cool} \rangle$ [Gyr]"),
        ("mean_tdyn_gyr", r"$\langle t_{\rm dyn} \rangle$ [Gyr]"),
    ]
):
    ax = axes[ax_i]
    finite = np.isfinite(rec_array("dominant_period")) & np.isfinite(rec_array(xkey))
    sc = ax.scatter(
        rec_array(xkey)[finite],
        rec_array("dominant_period")[finite],
        c=rec_array("sigma_osc")[finite],
        cmap="inferno",
        edgecolors="k",
        linewidths=0.4,
        s=50,
    )
    # 1:1 guide
    lims = [
        min(rec_array(xkey)[finite].min(), rec_array("dominant_period")[finite].min()),
        max(rec_array(xkey)[finite].max(), rec_array("dominant_period")[finite].max()),
    ]
    ax.plot(lims, lims, ls="--", color="grey", lw=1, label="1:1")
    ax.set(xlabel=xlabel, xscale="log", yscale="log")
    ax.legend(frameon=False, fontsize=10)

axes[0].set_ylabel(r"$P_{\rm osc}$ [Gyr]")
fig.suptitle("Oscillation period vs physical timescales", y=1.02)
cbar = fig.colorbar(sc, ax=axes, label=r"$\sigma_{\rm osc}$", shrink=0.85)
plt.tight_layout()
plt.savefig("figures/oscillation_period_vs_timescales.png", dpi=200, bbox_inches="tight")
plt.show()


# ──────────────────────────────────────────────────────────────────────
# PLOT 4: sigma_osc  vs  heating/cooling energy ratio
# ──────────────────────────────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
sc = ax.scatter(
    rec_array("heating_cooling_ratio"),
    rec_array("sigma_osc"),
    c=np.log10(rec_array("mhalo_zobs")),
    cmap="viridis",
    edgecolors="k",
    linewidths=0.4,
    s=50,
)
cbar = plt.colorbar(sc, ax=ax, label=r"$\log_{10} M_{\rm halo}(z_{\rm obs})$")
ax.set(
    xlabel=r"$\langle \dot{E}_{\rm wind} / \dot{E}_{\rm cool} \rangle$",
    ylabel=r"$\sigma_{\rm osc}$",
    xscale="log",
    title="Heating / cooling balance",
)
ax.axvline(1.0, ls="--", color="grey", lw=1, label="balance")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("figures/oscillation_sigma_vs_heating_cooling.png", dpi=200, bbox_inches="tight")
plt.show()

print("All phase-diagram plots saved to figures/")

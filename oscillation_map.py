# %%
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import cosmology
import astropy.units as u
import cgm_sf_regulator
from cgm_sf_regulator import CGMRegulator
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import astropy.constants as consts
from matplotlib.transforms import Affine2D
from cgm_sf_regulator import mhalo_at_z0_fakhouri

importlib.reload(cgm_sf_regulator)
plt.rcParams.update(
    {
        "text.usetex": False,
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
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

red1 = "tab:red"
red2 = "tab:orange"
blu1 = "dodgerblue"
blu2 = "tab:green"


def ln_fit(x, a, b):
    # fit log SFR vs time with a logarithmic function, after binning in 10 Myr
    return a * np.log(x) + b


def exp_fit(t, a, tau, c):
    return a * np.exp(t / tau) + c


# Gaussian model to fit counts vs residual (y)
def gaussian(x, A, mu, sig):
    return A * np.exp(-np.power((x - mu) / sig, 2.0) / 2)


# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)

sfr_bin_width = 0.01  # Gyr
ode_solver_step = 0.0005  # Gyr
residual_lim = (-0.06, 0.06)
spread_N_bins = 40
z_obs = 6.0
# %%
halo_masses_to_try = np.geomspace(1e9, 1e12, 20) * u.Msun
# mhalo_z0 = 1e12 * u.Msun
sigma_raw = []
sigma_fit = []
halo_mass_at_z_0 = []

for i, mhalo_zobs in enumerate(halo_masses_to_try[:1]):
    t_init = 0.1  # Gyr
    # get the final time given the observed redshift
    t_final = LCDM.age(z_obs).value  # Gyr
    print("** t_final = {:.2f}".format(t_final))
    t_span = (t_init, t_final)  # span of the integration

    mhalo_z0 = mhalo_at_z0_fakhouri(mhalo_zobs, z_obs) * u.Msun
    halo_mass_at_z_0.append(mhalo_z0.to(u.Msun).value)
    model_2Phase = CGMRegulator(
        mhalo_z0,
        t_span,
        tstep=ode_solver_step,  # in Gyr
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
    )
    run_2Phase = model_2Phase.run_halo()
    results_2Phase = model_2Phase.get_results()
    derived_2Phase = model_2Phase.get_derived_quantities()

    sfr = np.log10(derived_2Phase["dot_m_sfr"])

    t = results_2Phase["t"]  # in Gyr
    dot_m_cgm_cooling = np.log10(
        results_2Phase["m_cgm_hot"] / derived_2Phase["tcool_real"]
    )
    dot_m_cgm_in = np.log10(derived_2Phase["dot_m_cgm_in"])
    dot_m_sne_wind = np.log10(derived_2Phase["dot_m_ism_wind"])
    dot_m_cgm_ej = np.log10(derived_2Phase["dot_m_cgm_out"])
    dot_cgm_falling = np.log10(
        results_2Phase["m_cgm_cold"] / derived_2Phase["t_dynamical"]
    )
    dot_m_sfr = np.log10(derived_2Phase["dot_m_sfr"])

    dot_e_sne_wind = np.log10(derived_2Phase["dot_e_ism_wind"])
    dot_e_cgm_acc = np.log10(derived_2Phase["dot_e_cgm_in"])
    dot_e_cgm_cool = np.log10(derived_2Phase["dot_e_cgm_cooling"])
    dot_e_cgm_ej = np.log10(derived_2Phase["dot_e_cgm_out"])

    # instead of the rates, do the actual masses
    m_cgm = np.log10(results_2Phase["m_cgm"])
    m_ism = np.log10(results_2Phase["m_ism"])
    m_star = np.log10(results_2Phase["m_star"])

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(6, 5),
        sharex=True,
        dpi=300,
        gridspec_kw={"height_ratios": [1, 0.5]},
    )
    plt.subplots_adjust(hspace=0.05)

    xlim = t_span
    # use time (Gyr) on the bottom x-axis
    ax[0].plot(t, sfr, lw=2, label=r"$\dot{M}_{\rm \star}$", color="mediumorchid")

    # select finite, positive times
    mask = np.isfinite(t) & np.isfinite(sfr) & (t > 0)
    t_valid = t[mask]
    sfr_valid = sfr[mask]

    # bin in 10 Myr -> 0.01 Gyr

    log_sfr_min = 0.01 * np.max(sfr_valid)  # 1% of max SFR

    bins = np.arange(t.min(), t.max() + sfr_bin_width, sfr_bin_width)
    sfr_binned, bin_edges, _ = binned_statistic(
        t_valid, sfr_valid, statistic="mean", bins=bins
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # overplot binned points
    ax[0].scatter(
        bin_centers,
        sfr_binned,
        marker="o",
        s=10,
        edgecolor="k",
        facecolor="none",
        label=r"binned ({:.0f} Myr)".format(sfr_bin_width * 1000),
        zorder=5,
    )

    # keep only bins with finite means
    ok = np.isfinite(sfr_binned)
    t_binned = bin_centers[ok]
    sfr_binned = sfr_binned[ok]
    # fit within xlim and only above log mass

    fit_mask = (
        (t_binned >= xlim[0]) & (t_binned <= xlim[1]) & (sfr_binned >= log_sfr_min)
    )
    t_binned = t_binned[fit_mask]
    sfr_binned = sfr_binned[fit_mask]

    # # fit to the binned data
    # p0 = [-1.0, np.median(sfr_binned)]
    # popt, pcov = curve_fit(ln_fit, t_binned, sfr_binned, p0=p0, maxfev=10000)

    # do a exp fit as well
    p0_exp = [1.0, -0.5, np.median(sfr_binned)]
    popt_exp, pcov_exp = curve_fit(
        exp_fit, t_binned, sfr_binned, p0=p0_exp, maxfev=10000
    )

    ax[0].plot(
        t,
        exp_fit(t, *popt_exp),
        ls=":",
        lw=2,
        color="red",
        label="exp fit",
    )

    ax[0].set(
        ylabel=r"log mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        # ylim=(5, 12),
    )
    ax[0].xaxis.set_label_position("bottom")
    ax[0].xaxis.tick_bottom()

    # plot the residual of the fit and the SFR
    sfr_fit = exp_fit(t, *popt_exp)
    residual = np.log10(sfr / sfr_fit)

    ax[1].plot(t, residual, lw=2, color="gray", label="residuals")
    ax[1].axhline(0, ls="--", color="k", lw=1)
    ax[1].set(
        xlabel=r"time $[{\rm Gyr}]$",
        ylabel=r"log (SFR / fit SFR)",
        xlim=xlim,
        ylim=(residual_lim[0], residual_lim[1]),
    )
    # make a inset showing a histogram of the residuals and fit a Gaussian
    ax_inset = ax[1].inset_axes([1.1, 0, 0.35, 1])

    # bin the residuals
    hist_bins = np.linspace(residual_lim[0], residual_lim[1], spread_N_bins)
    counts, edges = np.histogram(residual, bins=hist_bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    std_residual = np.std(residual)  # XXX: precompute standard deviation of residuals

    # plot what is being fitted
    ax_inset.scatter(bin_centers, counts, marker="o", s=1, color="black", zorder=6)

    # also do a step plot for the histogram
    ax_inset.step(
        bin_centers, counts, where="mid", color="grey", linewidth=2, label="histogram"
    )

    #  fit Gaussian to the histogram of residuals
    p0 = [np.max(counts), 0, std_residual]
    popt_g, pcov_g = curve_fit(gaussian, bin_centers, counts, p0=p0)
    A, mu, sigma = popt_g

    # overlay fitted Gaussian (orientation horizontal: x=counts, y=residual)
    y_fit = np.linspace(residual_lim[0], residual_lim[1], 200)
    counts_fit = gaussian(y_fit, *popt_g)
    ax_inset.plot(y_fit, counts_fit, color="red", lw=2)

    ax_inset.set(xlim=(residual_lim[0], residual_lim[1]))
    # annotate fit parameters
    ax_inset.set_title(
        r"$\mu={:.1f}$,  $\sigma_{{\rm fit}}={:.6f}$, $\sigma_{{\rm est}}={:.6f}$".format(
            mu, sigma, std_residual
        ),
        fontsize=8,
    )

    # remove y tick labels
    ax_inset.set_yticklabels([])

    # Add twin redshift axis for the top row of plots (indices 0 and 1)
    # Twin redshift axis for top panel
    t_ticks = np.array([0.1, 0.3, 0.8, 1])
    z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
    ax2_top = ax[0].twiny()
    ax2_top.set(xlim=t_span)

    ax2_top.set_xticks(t_ticks)
    ax2_top.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2_top.set_xlabel(r"$z$", labelpad=8)
    ax2_top.minorticks_off()
    ax[0].minorticks_on()
    ax[0].legend(fontsize=9, loc="lower right")

    # label with mhalo at z=0
    ax[0].text(
        0.05,
        0.2,
        r"$M_{{\rm halo}}(z=0)={:.1e}$"
        "\n"
        r"$M_{{\rm halo}}(z_{{\rm obs}}={:.2f})={:.1e}\,{{\rm M_\odot }}$".format(
            mhalo_z0.to(u.Msun).value, z_obs, mhalo_zobs.to(u.Msun).value
        ),
        transform=ax[0].transAxes,
        fontsize=10,
    )
    # save
    sigma_raw.append(std_residual)
    sigma_fit.append(sigma)

    plt.savefig(
        f"figures/quantifying_sfr_oscillation_smaller_bins/{i:04d}_mhalo_{mhalo_z0.to(u.Msun).value:.1e}_Msun.png",
        bbox_inches="tight",
    )
    plt.show()


# %% check the f_prevent
halo_masses_to_try = np.geomspace(1e9, 1e12, 20) * u.Msun
z_obs = 6.0
# mhalo_z0 = 1e12 * u.Msun
sigma_raw = []
sigma_fit = []
halo_mass_at_z_0 = []

for i, mhalo_zobs in enumerate(halo_masses_to_try[:]):
    t_init = 0.1  # Gyr
    # get the final time given the observed redshift
    t_final = LCDM.age(z_obs).value  # Gyr
    print("** t_final = {:.2f}".format(t_final))
    t_span = (t_init, t_final)  # span of the integration

    mhalo_z0 = mhalo_at_z0_fakhouri(mhalo_zobs, z_obs) * u.Msun
    halo_mass_at_z_0.append(mhalo_z0.to(u.Msun).value)
    model_2Phase = CGMRegulator(
        mhalo_z0,
        t_span,
        tstep=ode_solver_step,  # in Gyr
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
    )
    run_2Phase = model_2Phase.run_halo()
    results_2Phase = model_2Phase.get_results()
    derived_2Phase = model_2Phase.get_derived_quantities()
    
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(5, 7),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 0.5, 0.5]},
    
    )
    halo_accretion = derived_2Phase["dot_m_cgm_in"] / derived_2Phase["f_prevent"]
    residual = (halo_accretion - derived_2Phase["dot_m_cgm_in"]) / halo_accretion
    residual_masked = residual[~np.isnan(residual)]
    residual_non_0 = residual_masked[residual_masked != 0]
    sigma_raw.append(np.std(residual_non_0  ))
    ax[0].plot(
        derived_2Phase["sim_time"],
        derived_2Phase["dot_m_cgm_in"],
        lw=2,
        label=r"$\dot{M}_{\rm cgm, in}$",
    )
    ax[0].plot(
        derived_2Phase["sim_time"],
        halo_accretion,
        lw=2,
        label=r"$f_{\rm baryon}\dot{M}_{halo}$",
    )
    ax[1].plot(
        derived_2Phase["sim_time"],
        residual,
        lw=2)


    ax[0].set(
        ylabel=r"$M_\odot / {\rm Gyr}$",
        xlim=(t_init, t_final),
        yscale="log",
    )
    ax[1].set(
        ylabel=r"residual",
        xlim=(t_init, t_final),
        # yscale="log",
        xlabel=r"time $[{\rm Gyr}]$",
    )
    ax[2].hist(residual_non_0, bins=np.linspace(0,1,30), color="gray", edgecolor="black", label=r"$\sigma$ ={:.6f}".format(np.std(residual_non_0)))
    ax[2].set(
        ylabel=r"counts",
        xlabel=r"residual",
    )
    ax[2].legend()
    ax[0].legend()
    plt.savefig(
    f"figures/quantifying_fprevent_oscillation/{i:04d}_mhalo_{mhalo_z0.to(u.Msun).value:.1e}_Msun.png",
    bbox_inches="tight",
)
    plt.show()
    #%% plot sigma as a function of halo mass
fig, ax = plt.subplots(figsize=(6, 4), dpi=300) 
ax.plot(halo_masses_to_try.to(u.Msun).value, sigma_raw, marker="o", ls="--")
ax.set(
    xlabel=r"$M_{\rm halo}(z=6) \, [{\rm M_\odot}]$",
    ylabel=r"$\sigma_{\rm raw}$",
    xscale="log",
)

# %% let's run a grid of models that also vary the loading factors and see how the oscillations change

halo_masses_to_try = np.geomspace(1e9, 1e12, 10) * u.Msun
# mhalo_z0 = 1e12 * u.Msun
sigma_raw = []
sigma_fit = []
alphaE_over_alphaM = []
halo_mass_at_z_0 = []
vir_temp_at_z_obs = []
cgm_temp_at_z_obs = []
halo_mass_observed = []
alpha_M = np.geomspace(1e-3, 1000, 10)
alpha_E = np.ones_like(alpha_M) * 1

for i, mhalo_zobs in enumerate(halo_masses_to_try[:]):

    for j, (aM, aE) in enumerate(zip(alpha_M, alpha_E)):

        # add a progress print
        print(
            "Running model {}/{}: mhalo_zobs = {:.2e} Msun, alpha_M = {:.2e}, alpha_E = {:.2e}".format(
                i * len(alpha_M) + j + 1,
                len(halo_masses_to_try) * len(alpha_M),
                mhalo_zobs.to(u.Msun).value,
                aM,
                aE,
            )
        )

        t_init = 0.1  # Gyr
        # get the final time given the observed redshift
        t_final = LCDM.age(z_obs).value  # Gyr
        print("** t_final = {:.2f}".format(t_final))
        t_span = (t_init, t_final)  # span of the integration

        mhalo_z0 = mhalo_at_z0_fakhouri(mhalo_zobs, z_obs) * u.Msun
        halo_mass_at_z_0.append(mhalo_z0.to(u.Msun).value)
        model_2Phase = CGMRegulator(
            mhalo_z0,
            t_span,
            tstep=ode_solver_step,  # in Gyr
            KS_kappa_s=0.1,
            add_f_prevent_floor=1e-6,
            verbose=False,
            alpha_m=aM,
            alpha_e=aE,
        )
        run_2Phase = model_2Phase.run_halo()
        results_2Phase = model_2Phase.get_results()
        derived_2Phase = model_2Phase.get_derived_quantities()
        Tvir_obs = derived_2Phase["halo_vir_temp"][-1]
        Tcgm_obs = derived_2Phase["cgm_temp"][-1]  # K

        sfr = np.log10(derived_2Phase["dot_m_sfr"])

        t = results_2Phase["t"]  # in Gyr
        dot_m_cgm_cooling = np.log10(
            results_2Phase["m_cgm_hot"] / derived_2Phase["tcool_real"]
        )
        dot_m_cgm_in = np.log10(derived_2Phase["dot_m_cgm_in"])
        dot_m_sne_wind = np.log10(derived_2Phase["dot_m_ism_wind"])
        dot_m_cgm_ej = np.log10(derived_2Phase["dot_m_cgm_out"])
        dot_cgm_falling = np.log10(
            results_2Phase["m_cgm_cold"] / derived_2Phase["t_dynamical"]
        )
        dot_m_sfr = np.log10(derived_2Phase["dot_m_sfr"])

        dot_e_sne_wind = np.log10(derived_2Phase["dot_e_ism_wind"])
        dot_e_cgm_acc = np.log10(derived_2Phase["dot_e_cgm_in"])
        dot_e_cgm_cool = np.log10(derived_2Phase["dot_e_cgm_cooling"])
        dot_e_cgm_ej = np.log10(derived_2Phase["dot_e_cgm_out"])

        # instead of the rates, do the actual masses
        m_cgm = np.log10(results_2Phase["m_cgm"])
        m_ism = np.log10(results_2Phase["m_ism"])
        m_star = np.log10(results_2Phase["m_star"])

        fig, ax = plt.subplots(
            2,
            1,
            figsize=(6, 5),
            sharex=True,
            dpi=300,
            gridspec_kw={"height_ratios": [1, 0.5]},
        )
        plt.subplots_adjust(hspace=0.05)

        xlim = t_span
        # use time (Gyr) on the bottom x-axis
        ax[0].plot(t, sfr, lw=2, label=r"$\dot{M}_{\rm \star}$", color="mediumorchid")

        # select finite, positive times
        mask = np.isfinite(t) & np.isfinite(sfr) & (t > 0)
        t_valid = t[mask]
        sfr_valid = sfr[mask]

        # bin in 10 Myr -> 0.01 Gyr

        log_sfr_min = 0.01 * np.max(sfr_valid)  # 1% of max SFR

        bins = np.arange(t.min(), t.max() + sfr_bin_width, sfr_bin_width)
        sfr_binned, bin_edges, _ = binned_statistic(
            t_valid, sfr_valid, statistic="mean", bins=bins
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # overplot binned points
        ax[0].scatter(
            bin_centers,
            sfr_binned,
            marker="o",
            s=10,
            edgecolor="k",
            facecolor="none",
            label=r"binned ({:.0f} Myr)".format(sfr_bin_width * 1000),
            zorder=5,
        )

        # keep only bins with finite means
        ok = np.isfinite(sfr_binned)
        t_binned = bin_centers[ok]
        sfr_binned = sfr_binned[ok]
        # fit within xlim and only above log mass

        fit_mask = (
            (t_binned >= xlim[0]) & (t_binned <= xlim[1]) & (sfr_binned >= log_sfr_min)
        )
        t_binned = t_binned[fit_mask]
        sfr_binned = sfr_binned[fit_mask]

        # # fit to the binned data
        # p0 = [-1.0, np.median(sfr_binned)]
        # popt, pcov = curve_fit(ln_fit, t_binned, sfr_binned, p0=p0, maxfev=10000)

        # do a exp fit as well
        p0_exp = [1.0, -0.5, np.median(sfr_binned)]
        popt_exp, pcov_exp = curve_fit(
            exp_fit, t_binned, sfr_binned, p0=p0_exp, maxfev=10000
        )

        ax[0].plot(
            t,
            exp_fit(t, *popt_exp),
            ls=":",
            lw=2,
            color="red",
            label="exp fit",
        )

        ax[0].set(
            ylabel=r"log mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
            # ylim=(5, 12),
        )
        ax[0].xaxis.set_label_position("bottom")
        ax[0].xaxis.tick_bottom()

        # plot the residual of the fit and the SFR
        sfr_fit = exp_fit(t, *popt_exp)
        residual = np.log10(sfr / sfr_fit)

        ax[1].plot(t, residual, lw=2, color="gray", label="residuals")
        ax[1].axhline(0, ls="--", color="k", lw=1)
        ax[1].set(
            xlabel=r"time $[{\rm Gyr}]$",
            ylabel=r"log (SFR / fit SFR)",
            xlim=xlim,
            ylim=(residual_lim[0], residual_lim[1]),
        )
        # make a inset showing a histogram of the residuals and fit a Gaussian
        ax_inset = ax[1].inset_axes([1.1, 0, 0.35, 1])

        # bin the residuals
        hist_bins = np.linspace(residual_lim[0], residual_lim[1], spread_N_bins)
        counts, edges = np.histogram(residual, bins=hist_bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        std_residual = np.std(
            residual
        )  # XXX: precompute standard deviation of residuals

        # plot what is being fitted
        ax_inset.scatter(bin_centers, counts, marker="o", s=1, color="black", zorder=6)

        # also do a step plot for the histogram
        ax_inset.step(
            bin_centers,
            counts,
            where="mid",
            color="grey",
            linewidth=2,
            label="histogram",
        )

        #  fit Gaussian to the histogram of residuals
        p0 = [np.max(counts), 0, std_residual]
        popt_g, pcov_g = curve_fit(gaussian, bin_centers, counts, p0=p0)
        A, mu, sigma = popt_g

        # overlay fitted Gaussian (orientation horizontal: x=counts, y=residual)
        y_fit = np.linspace(residual_lim[0], residual_lim[1], 200)
        counts_fit = gaussian(y_fit, *popt_g)
        ax_inset.plot(y_fit, counts_fit, color="red", lw=2)

        ax_inset.set(xlim=(residual_lim[0], residual_lim[1]))
        # annotate fit parameters
        ax_inset.set_title(
            r"$\mu={:.1f}$,  $\sigma_{{\rm fit}}={:.6f}$, $\sigma_{{\rm est}}={:.6f}$".format(
                mu, sigma, std_residual
            ),
            fontsize=8,
        )

        # remove y tick labels
        ax_inset.set_yticklabels([])

        # Add twin redshift axis for the top row of plots (indices 0 and 1)
        # Twin redshift axis for top panel
        t_ticks = np.array([0.1, 0.3, 0.8, 1])
        z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
        ax2_top = ax[0].twiny()
        ax2_top.set(xlim=t_span)

        ax2_top.set_xticks(t_ticks)
        ax2_top.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        ax2_top.set_xlabel(r"$z$", labelpad=8)
        ax2_top.minorticks_off()
        ax[0].minorticks_on()
        ax[0].legend(fontsize=9, loc="lower right")

        # label with mhalo at z=0
        ax[0].text(
            0.05,
            0.2,
            r"$M_{{\rm halo}}(z=0)={:.1e}$"
            "\n"
            r"$M_{{\rm halo}}(z_{{\rm obs}}={:.2f})={:.1e}\,{{\rm M_\odot }}$"
            "\n"
            r"$ \alpha_E / \alpha_M={:.6f}$".format(
                mhalo_z0.to(u.Msun).value, z_obs, mhalo_zobs.to(u.Msun).value, aE / aM
            ),
            transform=ax[0].transAxes,
            fontsize=10,
        )
        # save
        sigma_raw.append(std_residual)
        sigma_fit.append(sigma)
        alphaE_over_alphaM.append(aE / aM)
        vir_temp_at_z_obs.append(Tvir_obs)
        cgm_temp_at_z_obs.append(Tcgm_obs)
        halo_mass_at_z_0.append(mhalo_z0.to(u.Msun).value)
        halo_mass_observed.append(mhalo_zobs.to(u.Msun).value)
        vir_temp_at_z_obs.append(Tvir_obs)
        cgm_temp_at_z_obs.append(Tcgm_obs)
        plt.savefig(
            f"figures/quantifying_sfr_oscillation_grid/{i:02d}_{j:02d}_mhalo_{mhalo_z0.to(u.Msun).value:.1e}_Msun_alphaM_{aM:.1e}_alphaE_{aE:.1e}.png",
            bbox_inches="tight",
        )
        plt.show()
# %%
with open("runs/sfr_sigma_grid.txt", "w") as f:
    f.write(
        "# mhalo_obs[Msun]   sigma_raw  sigma_fit  alphaE/alphaM  vir_temp[K]  cgm_temp[K]\n"
    )
    for i in range(len(sigma_raw)):
        f.write(
            f"{halo_mass_observed[i]:.9e} "
            f"{sigma_raw[i]:.9e} {sigma_fit[i]:.9e} "
            f"{alphaE_over_alphaM[i]:.9e} "
            f"{vir_temp_at_z_obs[i]:.9e} {cgm_temp_at_z_obs[i]:.9e}\n"
        )
# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)


sc = ax.scatter(
    halo_mass_observed,
    alphaE_over_alphaM,
    c=sigma_raw,
    cmap="inferno",
    s=200,
    norm=matplotlib.colors.LogNorm(),
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"$\sigma_{\rm raw}$")
# cbar.set_label(r"$T_{\rm CGM} / T_{\rm vir}$ at $z=6$")
# cbar.set_label(r"$T_{\rm CGM} [K]$ at $z=6$")
ax.set(
    xlabel=r"$M_{\rm halo}(z=6) \, [{\rm M_\odot}]$",
    # xlabel=r"$T_{\rm vir} (z=6)\, [{\rm K}]$",
    ylabel=r"$\alpha_E / \alpha_M$",
    yscale="log",
    xscale="log",
)

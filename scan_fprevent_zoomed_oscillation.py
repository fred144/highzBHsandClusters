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
from astropy import constants as consts

importlib.reload(cgm_sf_regulator)
import pltstyle

plt.rcParams.update(
    {
        "text.usetex": False,
    }
)
red1 = "tab:red"
red2 = "tab:orange"
blu1 = "dodgerblue"
blu2 = "tab:green"


def vcirc_energy_loading(halo_vcirc, alpha_e=0.1):
    eta_e = alpha_e * (halo_vcirc.value / 200) ** (-3 / 2)

    # if eta e > 1 set to 1, halo_vcirc can be float or array
    if np.any(eta_e > 0):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def virial_T(mhalo, Rvir):

    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return (2 / 5) * ((G * mhalo * mp) / (Rvir * kb))


def vcirc_mass_loading(halo_vcirc, alpha_m=9):
    return alpha_m * (halo_vcirc.value / 200) ** (-3 / 2)


def virial_radius(z, mhalo, Delc=200):

    return (mhalo / (LCDM.critical_density(z) * (4 / 3) * np.pi * Delc)) ** (1 / 3)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def custom_mass_loading(mhalo, A=10, alpha=-1.4):
    """mass loading factor as a function of halo mass"""
    return A * (mhalo / (1e10 * u.solMass)) ** alpha


def custom_energy_loading(mhalo_z0, A=0.10, alpha=-0.5):
    """energy loading fact or as a function of halo mass"""
    eta_e = A * (mhalo_z0 / (1e12 * u.solMass)) ** alpha
    if np.any(eta_e > 1):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


mhalo_z0 = 1e13 * u.Msun
t_span = (0.1, 13.3)  # gyrs
xlim_for_zoom = (0.166, 2)

# %% do a single run of the 2 phase model and breakdown the energy and mass contributions
## this is figure 2 of the first draft of the SF paper

min_fprevent = np.linspace(0.01, 1, 20)  # we vary the minimum f_prevent value

for i, fp in enumerate(min_fprevent[::]):

    model_2phase = CGMRegulator(
        mhalo_z0,
        (0.2, xlim_for_zoom[1]),
        KS_kappa_s=0.1,
        add_f_prevent_floor=fp,
        verbose=False,
    )
    run_2phase = model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()

    model_2Phase_long = CGMRegulator(
        mhalo_z0, (0.15, 13), KS_kappa_s=0.1, add_f_prevent_floor=fp, verbose=False
    )
    run_2Phase_long = model_2Phase_long.run_halo()
    results_2Phase_long = model_2Phase_long.get_results()
    derived_2Phase_long = model_2Phase_long.get_derived_quantities()

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.23, wspace=0.2)

    # add some inset panels for long term evolution
    ax0 = ax[0].inset_axes([0, 1.2, 0.45, 1.4])
    ax1 = ax[1].inset_axes([0.55, 2.44, 0.45, 1.4])
    dot_m_cgm_cooling = (
        results_2Phase_long["m_cgm_hot"] / derived_2Phase_long["tcool_real"]
    )
    dot_m_cgm_in = derived_2Phase_long["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2Phase_long["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2Phase_long["dot_m_cgm_out"]
    dot_cgm_falling = (
        results_2Phase_long["m_cgm_cold"] / derived_2Phase_long["t_dynamical"]
    )
    dot_m_sfr = derived_2Phase_long["dot_m_sfr"]
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sfr,
        lw=2,
        label=r"$\dot{M}_{\rm \star}$",
        color="mediumorchid",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_ej,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, ej}$",
        color=blu2,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_cgm_falling,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, falling}$",
        color="y",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_cooling,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color=blu1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_in,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm CGM, acc}$",
        color=red1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sne_wind,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm SNe, wind}$",
        color=red2,
    )
    ax0.set(
        ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        yscale="log",
        ylim=(1e5, None),
        # xscale="log",
        xlabel=r"$z$",
        xlim=(0, 15),
    )
    # flip x axis
    ax0.invert_xaxis()

    # set the x axis to top
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.xaxis.set_tick_params(labeltop=True)
    ax0.xaxis.labelpad = 10

    dot_e_sne_wind = derived_2Phase_long["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2Phase_long["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2Phase_long["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2Phase_long["dot_e_cgm_out"]

    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_ej,
        color=blu2,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, ej}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_cool,
        color=blu1,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_acc,
        color=red1,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm CGM, acc}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_sne_wind,
        color=red2,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm SNe, wind}$",
    )
    ax1.set(
        ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
        yscale="log",
        ylim=(1e53, None),
        # xscale="log",
        xlabel=r"$z$",
        xlim=(0.0, 15),
    )
    # set the x axis to top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_tick_params(labeltop=True)
    ax1.invert_xaxis()
    # pad x label
    ax1.xaxis.labelpad = 10

    dot_m_cgm_cooling = results_2phase["m_cgm_hot"] / derived_2phase["tcool_real"]
    dot_m_cgm_in = derived_2phase["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2phase["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2phase["dot_m_cgm_out"]
    dot_cgm_falling = results_2phase["m_cgm_cold"] / derived_2phase["t_dynamical"]
    dot_m_sfr = derived_2phase["dot_m_sfr"]
    f_prevent = derived_2phase["dot_e_cgm_cooling"] / derived_2phase["dot_e_cgm_out"]

    red1 = "tab:red"
    red2 = "tab:orange"
    blu1 = "dodgerblue"
    blu2 = "tab:green"

    ax[0].plot(
        results_2phase["t"],
        dot_m_sfr,
        lw=3,
        label=r"$\dot{M}_{\rm \star}$",
        color="mediumorchid",
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_cgm_ej,
        lw=3,
        label=r"$\dot{M}_{\rm CGM, ej}$",
        color=blu2,
    )
    ax[0].plot(
        results_2phase["t"],
        dot_cgm_falling,
        lw=3,
        label=r"$\dot{M}_{\rm CGM, falling}$",
        color="y",
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_cgm_cooling,
        lw=3,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color=blu1,
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_cgm_in,
        ls="--",
        lw=3,
        label=r"$\dot{M}_{\rm CGM, acc}$",
        color=red1,
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_sne_wind,
        ls="--",
        lw=3,
        label=r"$\dot{M}_{\rm SNe, wind}$",
        color=red2,
    )
    ax[0].set(
        ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        yscale="log",
        ylim=(4e5, None),
        # xlabel=r"$t_{\rm univ}$ [Gyr]",
    )
    ax[0].legend(
        frameon=False,
        loc="upper center",
        fontsize=10.5,
        ncol=6,
        bbox_to_anchor=(0.5, 1.22),
    )

    # Panel 1: CGM energy rates
    dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
    dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool
    ax[1].plot(
        results_2phase["t"],
        dot_e_cgm_ej,
        color=blu2,
        lw=3,
        label=r"$\dot{E}_{\rm CGM, ej}$",
    )
    ax[1].plot(
        results_2phase["t"],
        dot_e_cgm_cool,
        color=blu1,
        lw=3,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
    )
    ax[1].plot(
        results_2phase["t"],
        dot_e_cgm_acc,
        color=red1,
        ls="--",
        lw=3,
        label=r"$\dot{E}_{\rm CGM, acc}$",
    )
    ax[1].plot(
        results_2phase["t"],
        dot_e_sne_wind,
        ls="--",
        color=red2,
        lw=3,
        label=r"$\dot{E}_{\rm SNe, wind}$",
    )

    ax[1].set(
        ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
        yscale="log",
        ylim=(5e52, None),
        xlim=xlim_for_zoom,
    )
    ax[1].legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=10.5,
    )
    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    # ===== add an inset axis to add a third row with the f_prevent evolution
    axin = ax[0].inset_axes([1.1, 1.6, 1, 1])
    axin.plot(
        results_2phase["t"],
        f_prevent,
        color="k",
        lw=3,
    )
    axin.set(
        ylabel=r"$f_{\rm prevent}$ unclipped",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        ylim=(5e-3, None),
        yscale="log",
    )
    axin.axhspan(fp, 1, color="k", alpha=0.1)

    # add another inset for the cgm temperature
    ax2 = ax[0].inset_axes([1.1, 0.26, 1, 1])
    ax2.plot(
        results_2phase["t"],
        derived_2phase["cgm_temp"],
        color="crimson",
        lw=3,
        label=r"$T_{\rm CGM}$",
    )

    # we can also plot the virial temperature for reference: halo_vir_temp
    ax2.plot(
        results_2phase["t"],
        derived_2phase["halo_vir_temp"],
        color="forestgreen",
        lw=3,
        ls="--",
        label=r"$T_{\rm vir}$",
    )
    ax2.legend(loc="lower right")

    ax2.set(
        ylabel=r"$\rm Temperature [K]$",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
    )
    axin.text(
        0.05,
        0.85,
        r"$f_{{\rm prevent, min}} = {:.6f}$".format(round(fp, 6)),
        transform=axin.transAxes,
        fontsize=12,
        ha="left",
        va="top",
        color="k",
    )

    # ======== add the lambda cooling rate in another inset
    ax3 = ax[0].inset_axes([1.1, -1, 1, 1])
    ax3.plot(
        results_2phase["t"],
        derived_2phase["cooling_lambda"],
        color="darkcyan",
        lw=3,
    )
    ax3.set(
        ylabel=r"$\Lambda$ [erg cm$^3$ s$^{-1}$]",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
        ylim=(1e-25, 1e-20),
    )

    # add the mass of the halo at z = 0on the first panel
    ax[0].text(
        0.95,
        0.05,
        r"$M_{{\rm halo}}(z=0) = {:.1e} \: M_{{\odot}}$".format(mhalo_z0.value),
        transform=ax[0].transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        color="k",
    )

    # add feedback parameters on the second panel
    zs = results_2Phase_long["z"]
    times = results_2Phase_long["t"] * u.Gyr
    mhalo_growth_new = results_2Phase_long["m_halo"] * u.Msun

    halo_rvir = virial_radius(zs, mhalo_growth_new).to(u.kpc)
    halo_vcirc = circular_velocity(mhalo_growth_new, halo_rvir).to(u.km / u.s)
    halo_vir_temp = virial_T(mhalo_growth_new, halo_rvir).to(u.K)

    eta_m = vcirc_mass_loading(halo_vcirc, alpha_m=0.1)
    eta_e = vcirc_energy_loading(halo_vcirc, alpha_e=0.1)

    ax4 = ax[1].inset_axes([1.1, -1.1, 1, 1])
    ax4.plot(times, eta_m, color=blu2, lw=3, label=r"$\eta_M$")
    ax4.plot(times, eta_e, color=red2, lw=3, label=r"$\eta_E$", ls="--")
    ax4.set(ylabel=r"loadings", yscale="log", xlim=xlim_for_zoom, ylim=(1e-3, 1e2))
    ax4.legend(loc="upper right")

    # show the metallicity inset
    ax5 = ax[1].inset_axes([0, -1.1, 1, 1])
    ax5.plot(
        results_2phase["t"],
        results_2phase["metal_cgm_mass_sol"],
        color="saddlebrown",
        lw=3,
    )
    ax5.set(
        ylabel=r"$Z_{\rm CGM} [Z_{\odot}]$",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
    )

    plt.savefig(
        "./figures/f_prevent_zoom_test_1e13/{:04d}_2phase_detailed_cgm_fp_{:}.png".format(
            i, round(fp, 6)
        ),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.show()

# %% same as above, what happens if we set a constant f_prevent

const_f_prevent_scan = np.linspace(0.1, 1, 10)

for fp in const_f_prevent_scan[::]:

    model_2phase = CGMRegulator(
        mhalo_z0,
        (0.2, xlim_for_zoom[1]),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        add_f_prevent_constant=fp,
    )
    model_2Phase_long = CGMRegulator(
        mhalo_z0,
        (0.15, 13),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        add_f_prevent_constant=fp,
    )
    run_2phase = model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()

    run_2Phase_long = model_2Phase_long.run_halo()
    results_2Phase_long = model_2Phase_long.get_results()
    derived_2Phase_long = model_2Phase_long.get_derived_quantities()

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.23, wspace=0.2)

    # add some inset panels for long term evolution
    ax0 = ax[0].inset_axes([0, 1.2, 0.45, 1.4])
    ax1 = ax[1].inset_axes([0.55, 2.44, 0.45, 1.4])
    dot_m_cgm_cooling = (
        results_2Phase_long["m_cgm_hot"] / derived_2Phase_long["tcool_real"]
    )
    dot_m_cgm_in = derived_2Phase_long["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2Phase_long["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2Phase_long["dot_m_cgm_out"]
    dot_cgm_falling = (
        results_2Phase_long["m_cgm_cold"] / derived_2Phase_long["t_dynamical"]
    )
    dot_m_sfr = derived_2Phase_long["dot_m_sfr"]
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sfr,
        lw=2,
        label=r"$\dot{M}_{\rm \star}$",
        color="mediumorchid",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_ej,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, ej}$",
        color=blu2,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_cgm_falling,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, falling}$",
        color="y",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_cooling,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color=blu1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_in,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm CGM, acc}$",
        color=red1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sne_wind,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm SNe, wind}$",
        color=red2,
    )
    ax0.set(
        ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        yscale="log",
        ylim=(1e5, None),
        # xscale="log",
        xlabel=r"$z$",
        xlim=(0, 15),
    )
    # flip x axis
    ax0.invert_xaxis()

    # set the x axis to top
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.xaxis.set_tick_params(labeltop=True)
    ax0.xaxis.labelpad = 10

    dot_e_sne_wind = derived_2Phase_long["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2Phase_long["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2Phase_long["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2Phase_long["dot_e_cgm_out"]

    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_ej,
        color=blu2,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, ej}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_cool,
        color=blu1,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_acc,
        color=red1,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm CGM, acc}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_sne_wind,
        color=red2,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm SNe, wind}$",
    )
    ax1.set(
        ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
        yscale="log",
        ylim=(5e54, None),
        # xscale="log",
        xlabel=r"$z$",
        xlim=(0.0, 15),
    )
    # set the x axis to top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_tick_params(labeltop=True)
    ax1.invert_xaxis()
    # pad x label
    ax1.xaxis.labelpad = 10

    dot_m_cgm_cooling = results_2phase["m_cgm_hot"] / derived_2phase["tcool_real"]
    dot_m_cgm_in = derived_2phase["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2phase["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2phase["dot_m_cgm_out"]
    dot_cgm_falling = results_2phase["m_cgm_cold"] / derived_2phase["t_dynamical"]
    dot_m_sfr = derived_2phase["dot_m_sfr"]
    f_prevent = derived_2phase["dot_e_cgm_cooling"] / derived_2phase["dot_e_cgm_out"]

    red1 = "tab:red"
    red2 = "tab:orange"
    blu1 = "dodgerblue"
    blu2 = "tab:green"

    ax[0].plot(
        results_2phase["t"],
        dot_m_sfr,
        lw=3,
        label=r"$\dot{M}_{\rm \star}$",
        color="mediumorchid",
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_cgm_ej,
        lw=3,
        label=r"$\dot{M}_{\rm CGM, ej}$",
        color=blu2,
    )
    ax[0].plot(
        results_2phase["t"],
        dot_cgm_falling,
        lw=3,
        label=r"$\dot{M}_{\rm CGM, falling}$",
        color="y",
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_cgm_cooling,
        lw=3,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color=blu1,
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_cgm_in,
        ls="--",
        lw=3,
        label=r"$\dot{M}_{\rm CGM, acc}$",
        color=red1,
    )
    ax[0].plot(
        results_2phase["t"],
        dot_m_sne_wind,
        ls="--",
        lw=3,
        label=r"$\dot{M}_{\rm SNe, wind}$",
        color=red2,
    )
    ax[0].set(
        ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        yscale="log",
        ylim=(4e5, None),
        # xlabel=r"$t_{\rm univ}$ [Gyr]",
    )
    ax[0].legend(
        frameon=False,
        loc="upper center",
        fontsize=10.5,
        ncol=6,
        bbox_to_anchor=(0.5, 1.22),
    )

    # Panel 1: CGM energy rates
    dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
    dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool
    ax[1].plot(
        results_2phase["t"],
        dot_e_cgm_ej,
        color=blu2,
        lw=3,
        label=r"$\dot{E}_{\rm CGM, ej}$",
    )
    ax[1].plot(
        results_2phase["t"],
        dot_e_cgm_cool,
        color=blu1,
        lw=3,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
    )
    ax[1].plot(
        results_2phase["t"],
        dot_e_cgm_acc,
        color=red1,
        ls="--",
        lw=3,
        label=r"$\dot{E}_{\rm CGM, acc}$",
    )
    ax[1].plot(
        results_2phase["t"],
        dot_e_sne_wind,
        ls="--",
        color=red2,
        lw=3,
        label=r"$\dot{E}_{\rm SNe, wind}$",
    )

    ax[1].set(
        ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
        yscale="log",
        ylim=(5e52, None),
        xlim=xlim_for_zoom,
    )
    ax[1].legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=10.5,
    )
    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    # ===== add an inset axis to add a third row with the f_prevent evolution
    axin = ax[1].inset_axes([0, -1.2, 1, 1])
    axin.plot(
        results_2phase["t"],
        f_prevent,
        color="k",
        lw=3,
    )
    axin.set(
        ylabel=r"$f_{\rm prevent}$ constant",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        ylim=(5e-3, None),
        yscale="log",
    )

    # add another inset for the cgm temperature
    ax2 = ax[1].inset_axes([0, -2.22, 1, 1])
    ax2.plot(
        results_2phase["t"],
        derived_2phase["cgm_temp"],
        color="crimson",
        lw=3,
        label=r"$T_{\rm CGM}$",
    )

    # we can also plot the virial temperature for reference: halo_vir_temp
    ax2.plot(
        results_2phase["t"],
        derived_2phase["halo_vir_temp"],
        color="forestgreen",
        lw=3,
        ls="--",
        label=r"$T_{\rm vir}$",
    )
    ax2.legend(loc="lower right")

    ax2.set(
        ylabel=r"$\rm Temperature [K]$",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
    )
    axin.text(
        0.05,
        0.85,
        r"$f_{{\rm prevent, constant}} = {:.6f}$".format(round(fp, 6)),
        transform=axin.transAxes,
        fontsize=12,
        ha="left",
        va="top",
        color="k",
    )

    # add the lambda cooling rate in another inset
    ax3 = ax[1].inset_axes([0, -3.24, 1, 1])
    ax3.plot(
        results_2phase["t"],
        derived_2phase["cooling_lambda"],
        color="darkcyan",
        lw=3,
    )
    ax3.set(
        ylabel=r"$\Lambda$ [erg cm$^3$ s$^{-1}$]",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
        ylim=(1e-25, 1e-20),
    )

    # add the mass of the halo at z = 0on the first panel
    ax[0].text(
        0.05,
        0.9,
        r"$M_{{\rm halo}}(z=0) = {:.1e} \: M_{{\odot}}$".format(mhalo_z0.value),
        transform=ax[0].transAxes,
        fontsize=12,
        ha="left",
        va="top",
        color="k",
    )
    plt.savefig(
        "./figures/f_prevent_zoom_test_1e12/2phase_detailed_cgm_1e12_fp_{:.6f}.png".format(
            fp
        ),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.show()

# %%
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import cosmology
import astropy.units as u
import cgm_sf_regulator
from cgm_sf_regulator import CGMRegulator, CoolingFunctionInterpolator
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import astropy.constants as consts
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

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


# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 13.3)  # gyrs
xlim_for_zoom = (0.166, 2)


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


# %% #test out the CoolingFunctionInterpolator


# %%
# add this function near the top of your file (before the loop)
def plot_cooling_table_scan(
    results_2phase,
    derived_2phase,
    results_2Phase_long,
    derived_2Phase_long,
    T_thresh_cool,
    T_slope,
    mhalo_z0,
    xlim_for_zoom=(0.166, 2),
    show=True,
):

    red1 = "tab:red"
    red2 = "tab:orange"
    blu1 = "dodgerblue"
    blu2 = "tab:green"

    # long-run mass/energy rates
    dot_m_cgm_cooling_long = (
        results_2Phase_long["m_cgm_hot"] / derived_2Phase_long["tcool_real"]
    )
    dot_m_cgm_in_long = derived_2Phase_long["dot_m_cgm_in"]
    dot_m_sne_wind_long = derived_2Phase_long["dot_m_ism_wind"]
    dot_m_cgm_ej_long = derived_2Phase_long["dot_m_cgm_out"]
    dot_cgm_falling_long = (
        results_2Phase_long["m_cgm_cold"] / derived_2Phase_long["t_dynamical"]
    )
    dot_m_sfr_long = derived_2Phase_long["dot_m_sfr"]

    dot_e_sne_wind_long = derived_2Phase_long["dot_e_ism_wind"]
    dot_e_cgm_acc_long = derived_2Phase_long["dot_e_cgm_in"]
    dot_e_cgm_cool_long = derived_2Phase_long["dot_e_cgm_cooling"]
    dot_e_cgm_ej_long = derived_2Phase_long["dot_e_cgm_out"]

    # short-run (zoom) mass/energy rates and derived quantities
    dot_m_cgm_cooling = results_2phase["m_cgm_hot"] / derived_2phase["tcool_real"]
    dot_m_cgm_in = derived_2phase["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2phase["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2phase["dot_m_cgm_out"]
    dot_cgm_falling = results_2phase["m_cgm_cold"] / derived_2phase["t_dynamical"]
    dot_m_sfr = derived_2phase["dot_m_sfr"]
    f_prevent = derived_2phase["dot_e_cgm_cooling"] / derived_2phase["dot_e_cgm_out"]

    dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
    dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool

    # use the long results to compute feedback properties
    zs = results_2Phase_long["z"]
    times = results_2Phase_long["t"] * u.Gyr
    mhalo_growth_new = results_2Phase_long["m_halo"] * u.Msun

    halo_rvir = virial_radius(zs, mhalo_growth_new).to(u.kpc)
    halo_vcirc = circular_velocity(mhalo_growth_new, halo_rvir).to(u.km / u.s)
    halo_vir_temp = virial_T(mhalo_growth_new, halo_rvir).to(u.K)

    eta_m = vcirc_mass_loading(halo_vcirc, alpha_m=0.1)
    eta_e = vcirc_energy_loading(halo_vcirc, alpha_e=0.1)

    # old mass loadings
    # eta_m = custom_mass_loading(mhalo_growth_new, A=10, alpha=-0.7)
    # eta_e = custom_energy_loading(mhalo_growth_new, A=0.1, alpha=-0.5)

    # create figure and main axes
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=300, gridspec_kw={'height_ratios': [1, 1, 2]})
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # long-run insets at top of each panel
    ax0 = ax[0].inset_axes([0, 1.2, 0.45, 1.4])
    ax1 = ax[0].inset_axes([0.55, 1.2, 0.45, 1.4])

    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sfr_long,
        lw=2,
        label=r"$\dot{M}_{\rm \star}$",
        color="mediumorchid",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_ej_long,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, ej}$",
        color=blu2,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_cgm_falling_long,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, falling}$",
        color="y",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_cooling_long,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color=blu1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_in_long,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm CGM, acc}$",
        color=red1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sne_wind_long,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm SNe, wind}$",
        color=red2,
    )
    ax0.set(
        ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        yscale="log",
        ylim=(1e7, None),
        xlabel=r"$z$",
        xlim=(0, 15),
    )
    ax0.invert_xaxis()
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.xaxis.set_tick_params(labeltop=True)
    ax0.xaxis.labelpad = 10

    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_ej_long,
        color=blu2,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, ej}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_cool_long,
        color=blu1,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_acc_long,
        color=red1,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm CGM, acc}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_sne_wind_long,
        color=red2,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm SNe, wind}$",
    )
    ax1.set(
        ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
        yscale="log",
        ylim=(5e54, None),
        xlabel=r"$z$",
        xlim=(0.0, 15),
    )
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_tick_params(labeltop=True)
    ax1.invert_xaxis()
    ax1.xaxis.labelpad = 10

    # main short/zoom panels
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
        xlim=xlim_for_zoom,
    )
    ax[0].legend(
        frameon=False,
        loc="upper center",
        fontsize=10.5,
        ncol=6,
        bbox_to_anchor=(0.5, -0.1),
       
    )

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
        bbox_to_anchor=(0.5, -0.1),
        fontsize=10.5,
    )

    # ensure lines sit behind axes labels
    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    # f_prevent inset
    axin = ax[0].inset_axes([1.1, 1.75, 1, 1])
    axin.plot(results_2phase["t"], f_prevent, color="k", lw=3)
    axin.set(
        ylabel=r"$f_{\rm prevent}$ unclipped",
        xlim=xlim_for_zoom,
        ylim=(5e-3, None),
        yscale="log",
    )
    axin.axhspan(1e-6, 1, color="k", alpha=0.1)

    # CGM temperature inset
    ax2 = ax[1].inset_axes([1.1, 1.85, 1, 1])
    ax2.plot(
        results_2phase["t"],
        derived_2phase["cgm_temp"],
        color="crimson",
        lw=3,
        label=r"$T_{\rm CGM}$",
    )
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
        xlim=xlim_for_zoom,
        yscale="log",
    )

    # cooling lambda inset
    ax3 = ax[1].inset_axes([1.1, 0.6, 1, 1])
    ax3.plot(
        results_2phase["t"], derived_2phase["cooling_lambda"], color="darkcyan", lw=3
    )
    ax3.set(
        ylabel=r"$\Lambda$ [erg cm$^3$ s$^{-1}$]",
        xlim=xlim_for_zoom,
        yscale="log",
        ylim=(1e-25, 1e-20),
    )

    # halo mass annotation
    ax[0].text(
        0.5,
        0.1,
        r"$M_{{\rm halo}}(z=0) = {:.1e} \: M_{{\odot}}$".format(mhalo_z0.value),
        transform=ax[0].transAxes,
        fontsize=12,
        ha="center",
        va="bottom",
        color="k",
    )
   

    # if show:
    #     plt.show()
    # loading inset
    ax4 = ax[1].inset_axes([1.1, -0.8, 1, 1])
    ax4.plot(times, eta_m, color=blu2, lw=3, label=r"$\eta_M$")
    ax4.plot(times, eta_e, color=red2, lw=3, label=r"$\eta_E$", ls="--")
    ax4.set(ylabel=r"loadings", yscale="log", xlim=xlim_for_zoom, ylim=(1e-3, 1e2))
    ax4.legend(loc="upper right")

    # show the metallicity inset
    ax5 = ax[1].inset_axes([1.1, -2.2, 1, 1])
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
    # add the cooling tables
    temps = np.geomspace(1e4, 1e8, 100)
    metallicities = np.geomspace(1e-3, 2, 10)
    sutherland_dopita = CoolingFunctionInterpolator(
        file_path="./tables/newcool_viraj.dat"
    )

    N = len(metallicities)
    # pick a continuous cmap and make N discrete colors
    base_cmap = plt.get_cmap("coolwarm_r")
    colors = base_cmap(np.linspace(0, 1, N))
    lcmap = ListedColormap(colors)
    bounds = np.arange(N + 1)
    norm = BoundaryNorm(bounds, lcmap.N)

    for i, Z in enumerate(metallicities):

        lambda_cool_custom = sutherland_dopita.custom_cooling_function(
            temps, Z, T_thresh_cool=T_thresh_cool, T_slope=T_slope
        ) * (u.erg * u.cm**3 * u.s**-1)
        ax[2].plot(
            temps, lambda_cool_custom, color=lcmap(i), label=f"Z={Z:.1e} Zsun", lw=1.5
        )

    ax[2].set(
        xscale="log",
        yscale="log",
        xlabel="CGM Temperature [K]",
        ylabel=r"$\Lambda$ [erg cm$^3$ s$^{-1}$]",
    )

    # segmented colorbar in an inset axis (easy positioning)
    cax = ax[2].inset_axes([0.6, 0.2, 0.3, 0.1])
    # use a log norm for the colorbar spanning the metallicity range
    vmin = metallicities.min()
    vmax = metallicities.max()
    mappable = mpl.cm.ScalarMappable(
        cmap=lcmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    mappable.set_array(metallicities)
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    # remove numeric tick labels
    # cbar.set_ticks([])
    cbar.set_label("metallicity [Z$_\\odot$]")


    return fig, ax


# %% do a single run of the 2 phase model and breakdown the energy and mass contributions
## this is figure 2 of the first draft of the SF paper


# m_loadings = np.linspace(0.12, 10, 20) #np.linspace(0.1, 10, 5)
# e_loadings = np.ones_like(m_loadings) * 0.1

# low specific energy to high specific energy
T_slope = -np.linspace(0.1,1, 40)  # dimensionless
threshold_for_pwrlaw_cooling = 1e4 * np.ones_like(T_slope)  # K

for i, (T_th, T_s) in enumerate(zip(threshold_for_pwrlaw_cooling, T_slope)):

    model_2phase = CGMRegulator(
        mhalo_z0,
        (0.2, xlim_for_zoom[1]),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        # alpha_e=0.1,
        # alpha_m=0.1,
        updated_loadings=True,
        updated_halo_infall=True,
        dbug_norm_for_2_phase_CGM=0,
        custom_cooling_params=(T_th, T_s)
    )

    model_2Phase_long = CGMRegulator(
        mhalo_z0,
        (0.15, 13),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        # alpha_e=0.1,
        # alpha_m=0.1,
        updated_loadings=True,
        updated_halo_infall=True,
        dbug_norm_for_2_phase_CGM=0,
        custom_cooling_params=(T_th, T_s)
    )

    run_2phase = model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()

    run_2Phase_long = model_2Phase_long.run_halo()
    results_2Phase_long = model_2Phase_long.get_results()
    derived_2Phase_long = model_2Phase_long.get_derived_quantities()

    fig, ax = plot_cooling_table_scan(
        results_2phase,
        derived_2phase,
        results_2Phase_long,
        derived_2Phase_long,
        T_thresh_cool=T_th,
        T_slope=T_s,
        mhalo_z0=mhalo_z0,
        xlim_for_zoom=xlim_for_zoom,
        show=False,
    )

    plt.savefig(
        f"./figures/scan_custom_cooling_curve_slope_negative/{i:04d}_Tth_{T_th:.4e}_Tslope_{T_s:.2f}_coolingtable_scan.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()

# %%

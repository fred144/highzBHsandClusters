# %% need to see what the results of varying the normalization kappa in the KS law
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmasher as cmr
import astropy.units as u
from astropy import constants as consts
from astropy import cosmology
from cgm_sf_regulator import CGMRegulator
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
import os
from astropy.table import Table
import pandas as pd
from matplotlib.lines import Line2D
from labellines import labelLine, labelLines
from cgm_sf_regulator import CGMRegulator
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        # "text.usetex": True,
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
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)
# %%
# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


# calculate corresponding virial temperatures
def virial_T(mhalo, Rvir):
    """Halo fp

    Args:
        mhalo (_type_): halo mass
        Rvir (_type_): halo virial radius

    Returns:
        _type_: virial temperture
    """
    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return ((2 / 5) * ((G * mhalo * mp) / (Rvir * kb))).to(u.K)


def virial_radius(z, mhalo, Delc=200):
    """
    Halo virial radius, classical 200 top-hat overdensity.
    Supports z as a 1D array and mhalo as a 2D array (shape: [len(z), N]).
    Args:
        z (array-like): Redshift array of shape (len(z),)
        mhalo (array-like): Halo mass array of shape (len(z), N)
        Delc (float): Overdensity parameter (default: 200)
    Returns:
        ndarray: Virial radius array of shape (len(z), N)
    """
    z = np.asarray(z)

    # Broadcast critical density to shape (len(z), 1)
    rhoc = LCDM.critical_density(z)[:, np.newaxis]
    rvir = (mhalo / (rhoc * (4 / 3) * np.pi * Delc)) ** (1 / 3)

    return rvir.to(u.kpc)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def vcirc_from_virial_T(Tvir, mu=0.59):
    kb = consts.k_B
    mp = consts.m_p
    return np.sqrt((2 * kb * Tvir) / (mu * mp)).to(u.km / u.s)


zbins_str = [
    "0.2 < z < 0.5",
    # "0.5 < z < 0.8",
    # "0.8 < z < 1.1",
    # "1.1 < z < 1.5",
    # "1.5 < z < 2.0",
    "2.0 < z < 2.5",
    "2.5 < z < 3.0",
    # "3.0 < z < 3.5",
    "3.5 < z < 4.5",
    # "4.5 < z < 5.5",
    "5.5 < z < 6.5",
    "6.5 < z < 7.5",
    "7.5 < z < 8.5",
    # "8.5 < z < 10.0",
    "10.0 < z < 12.0",
]

# make a color map with the same number of colors as the number of zbins_str
cmap = cmr.tropical
cmap_colors = [cmap(i / len(zbins_str)) for i in range(len(zbins_str))]

zbins_ctr = []  # get the center value of zbins_str
for zb in zbins_str:
    z = zb.split("<")
    z = (float(z[0]) + float(z[2])) / 2
    zbins_ctr.append(z)


smf_data = Table.read("./data/Shuntov2024-shmr.ecsv", format="ascii.ecsv")

# define empty dictionaries to store values for each z-bin
Mhalo = {}
Mstar = {}
Mstar_low = {}
Mstar_up = {}

# read for each z-bin
for zb in zbins_str:
    Mhalo[zb] = smf_data[smf_data["Redshift"] == zb]["M_halo"]
    Mstar[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_50"]
    Mstar_low[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_16"]
    Mstar_up[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_84"]

# reverse zbins_ctr to match the order of the data
zbins_ctr = zbins_ctr[::-1]
zbins_ctr.append(0.01)  # this is the lowest redshift bin

# %% this if looping through KS parameters for param variation studies
mass_bins = 10
zobs = zbins_ctr
# make a unique halo array for each redshift
mhalos = np.geomspace(1e10, 1e13, mass_bins)
mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun
print(zobs)
print(mhalos[0])

# kappa_sfr = 0.1
kappa_sfrs = [0.02]#np.geomspace(0.01, 10, 10)[::-1]
# kappa_sfr = 1
# n_sfrs = np.linspace(1.3, 1.5, 10)
# r_disk_sfrs = np.geomspace(0.011, 0.1, 15)
r_disk_sfr = 0.018
# n_sfrs = [1.4]
n_sfrs = [1.8] #np.linspace(1.1, 2.0, 10)
# kappa_sfr = 1.0
for j, n_sfr in enumerate(n_sfrs):
    
    # varying both, iterate kappa_sfr and n_sfr together
    kappa_sfr = kappa_sfrs[j]
    
    param_txt = (
    f"KS_1998_kappa{str(kappa_sfr).replace('.', 'p')}_"
    + f"n{str(n_sfr).replace('.', 'p')}_"
    + f"r{str(r_disk_sfr).replace('.', 'p')}"
)
    file = "./runs/smhm_2phase_redshift_scan_" f"{param_txt}.h5"

    ####

    if not os.path.exists(file):
        print("running 2-phase model grid...", file)
        redshift_variation, zsims = run_2phase_model_redshift_grid(
            observe_at=zobs,  # redshift we want to observe
            mhalos=mhalos,
            write_to_file=file,
            disk_scale_length=r_disk_sfr,
            KS_n=n_sfr,
            KS_kappa_s=kappa_sfr,
            # add_f_prevent_floor=1e-6
            KS_parametrization = "KS1998"
        )


    f = h5py.File(file, "r")
    smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
    mhalo_obs = f["Mhalo_obs"]
    mstar = f["Mstar_obs"]
    mism = f["MISM_obs"]
    zs = f["redshifts"][:]
    sfr = f["SFR_obs"]
    mmetal_cgm = f["MMetals_obs"]
    rvirs = virial_radius(zs, mhalo_obs * u.Msun)  # virial radius in kpc
    halo_Tvirs = virial_T(mhalo_obs * u.Msun, rvirs)  # virial temperature in K
    print(f.keys())
    print(f["redshifts"][:])


    fig, ax = plt.subplots(2, 1, figsize=(5, 6.5), dpi=300, sharex=True, sharey="row")
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.05)
    # get a colormap from cmasher
    cmap = plt.get_cmap("cmr.tropical")
    colors = cmap(np.linspace(0, 1, len(zs)))
    # replace the final color with grey for the lowest redshift
    colors[-1] = [0.5, 0.5, 0.5, 1.0]
    Tvir_max = 1e6 * u.K
    # loop through redshift and plot mgas/mstar vs mstar
    for i, z in enumerate(zs):
        Tvirs = halo_Tvirs[i]
        mask = Tvirs < Tvir_max

        ax[0].plot(
            mstar[i],
            mism[i] / mstar[i],
            color=colors[i],
            label=r"${:.1f}$".format(z),
            lw=3
        )
        ax[0].scatter(
            mstar[i][mask],
            mism[i][mask] / mstar[i][mask],
            s=30,
            color=colors[i],
            edgecolor="k",
            zorder=3,
        )

        t_depletion = mism[i] / sfr[i]
        sSFR_Gyr = sfr[i] / mstar[i]  # Gyr^-1
        sSFR_yr = sSFR_Gyr * 1e-9
        sfr_yr = sfr[i] * 1e-9
        ax[1].plot(
            mstar[i],
            sfr_yr,
            color=colors[i],
            lw=3
        )
        ax[1].scatter(
            mstar[i][mask], sfr_yr[mask], s=30, color=colors[i], edgecolor="k", zorder=3
        )

    lines = ax[0].get_lines()
    l1 = lines[0]
    labelLine(
        l1,
        2e9,
        label=r"$z={}$".format(zs[0]),
        align=False,
        yoffset=0.0,
        ha="left",
        backgroundcolor="none",
        fontsize=7,
        # color="black",
    )
    labelLines(
        lines[1:],
        xvals=np.geomspace(1e9, 1e12, len(lines) - 1),
        yoffsets=0.01,
        align="right",
        fontsize=7,
        ha="right",
        backgroundcolor="none",
        # color="black",
    )


    # make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right

    # # Calculate the slope and intercept in log-log space
    # x1, y1 = 8.7, 0.2
    # x2, y2 = 10.35, -0.4

    # # Generate x values spanning the plot range
    # x_vals = np.linspace(6, 12, 100)
    # # Compute corresponding y values for the line in log-log space
    # slope = (y2 - y1) / (x2 - x1)
    # y_vals = y1 + slope * (x_vals - x1)

    # ax[0].plot(
    #     10**x_vals,
    #     10**y_vals,
    #     ls="-",
    #     color="k",
    #     lw="4",
    #     alpha=0.5,
    #     zorder=1,
    # )
    # # put a text label near this line
    # ax[0].text(
    #     0.85,
    #     0.22,
    #     r"Calette+18, $z\sim 0$",
    #     transform=ax[0].transAxes,
    #     fontsize=8,
    #     rotation=-35,
    #     va="center",
    #     ha="right",
    # )

    ### ============== observations for comparison ==============

    ## xGass: Catinella et al: https://ui.adsabs.harvard.edu/abs/2018MNRAS.476..875C/abstract

    # total Mgas/Mstar vs Mstar, table 2 average values per bin
    catinella_mstar = 10 ** np.array([9.16, 9.44, 9.75, 10.05, 10.34, 10.65, 10.95, 11.21])
    catinella_mgas_mstar = 10 ** np.array(
        [0.098, -0.136, -0.509, -0.518, -0.817, -0.958, -1.190, -1.328]
    )
    catinella_mgas_mstar_err = np.array(
        [0.064, 0.077, 0.076, 0.062, 0.055, 0.048, 0.048, 0.064]
    )
    yerr_lower = catinella_mgas_mstar * (1 - 10 ** (-catinella_mgas_mstar_err))
    yerr_upper = catinella_mgas_mstar * (10 ** (catinella_mgas_mstar_err) - 1)
    yerr = np.vstack([yerr_lower, yerr_upper])

    # here for the raw counts, unbinnned unlike above, using the full table
    # https://xgass.icrar.org/assets/data/xGASS_representative_sample.readme
    # https://xgass.icrar.org/assets/data/xGASS_representative_sample.ascii
    # 0.01 < z < 0.05 Catinella+18 still, but for some reason don't have full gas fraction so using binned values
    xGass_dat = pd.read_csv(
        "./data/xGass.txt",
        delim_whitespace=True,
    )
    xgas_sfrbest = xGass_dat["SFR_best"]  # msun /year
    xgas_mstar = 10 ** xGass_dat["lgMstar"]
    detection_tag = xGass_dat["HIsrc"]  # 1 for detection, 0 for upper limit
    alfalfa_tag = 1
    mask = detection_tag == alfalfa_tag  # only the gas rich alfalfa detections
    ax[0].errorbar(
        catinella_mstar,
        catinella_mgas_mstar,
        yerr=yerr,
        fmt="o",
        color="k",
        markersize=3,
        alpha=0.5,
        elinewidth=1,
    )
    ax[1].errorbar(
        xgas_mstar[mask],
        np.array(xgas_sfrbest[mask]),
        color="k",
        markersize=3,
        fmt="o",
        alpha=0.3,
        zorder=1,
    )

    # ssfr fit Steven Janowieck et al. 2020, xGASS: cold gas content and quenching in galaxies below the star-forming main sequence Eq https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.1982J/abstract, eq 1
    # m_sfms = -0.344 # +/- 0.101
    # b_sfrms = -9.822 # +/- 0.057
    # m_sfs = 0.088 # +/- 0.028
    # b_sfs = 0.188 # +/- 0.036
    # m_star_test = np.log10(np.geomspace(1e9, 1e11, 100))
    # log_sFR_ms = m_sfms * (m_star_test - 9) + b_sfrms
    # # now translate to sSFR for plotting in panel 1
    # sSFR_ms = 10**log_sFR_ms
    # ax[1].plot(10**m_star_test, sSFR_ms, ls='--', color='k', label='xGASS SFMS (Janowiecki+20)')


    ## https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.1526P/abstract Popesso eneded up no using


    ## Callette relation https://arxiv.org/pdf/1803.07692, late type galaxy Rgas-Mstar
    log_C_ltg = 4.76
    log_C_err = 0.05
    a_ltg = -0.52
    a_err = 0.03
    intrinsic_scatter_dex = 0.44

    ### early type galaxies
    log_C_etg = 3.70
    a_etg = -0.58
    intrinsic_scatter_dex_etg = 0.68

    # Plot Calette relation: y(M*) = C * (M*/M_sun)^a
    log_m_star_thry = np.linspace(6, 12, 100)
    m_star_thry = 10**log_m_star_thry

    # y_catinella_etg = 10**log_C_etg * (m_star_thry)**a_etg
    # ax[0].plot(m_star_thry, y_catinella_etg, ls='--', color='k', lw=2, label='Callette+18 ETG', zorder=2)

    # now the late type galaxies
    y_catinella_ltg = 10**log_C_ltg * (m_star_thry) ** a_ltg
    scatter_ltg = 0.44

    y_upper = y_catinella_ltg * 10**scatter_ltg
    y_lower = y_catinella_ltg / 10**scatter_ltg
    # ax[0].plot(
    #     m_star_thry,
    #     y_catinella_ltg,
    #     ls="-",
    #     color="k",
    #     lw=2,
    #     label="Callette+18 LTG",
    #     zorder=0,
    # )
    # ax[0].fill_between(m_star_thry, y_lower, y_upper, facecolor='k', alpha=0.2, zorder=0)

    # Calette but double power law, still LTG
    C = 1.69
    a = 0.18
    b = 0.61
    log_Mtrunc = 9.2
    intrinsic_scatter_dbl_pl = 0.44
    y_dbl_power = C / (
        (m_star_thry / 10**log_Mtrunc) ** (a) + (m_star_thry / 10**log_Mtrunc) ** b
    )
    ax[0].plot(
        m_star_thry,
        y_dbl_power,
        ls="-.",
        color="thistle",
        lw=2,
        label="Calette+18 LTG DPL",
        zorder=0,
    )
    y_dbl_power_upper = y_dbl_power * 10**intrinsic_scatter_dbl_pl
    y_dbl_power_lower = y_dbl_power / 10**intrinsic_scatter_dbl_pl
    ax[0].fill_between(
        m_star_thry,
        y_dbl_power_lower,
        y_dbl_power_upper,
        facecolor="thistle",
        alpha=0.5,
        zorder=0,
    )


    x_theory = np.geomspace(1e6, 1e12, 20)  # some theory lines


    # Renzini & Peng 2015 SFMS with 0.3 dex scatter
    def logSFR_with_errors(
        Mstar, slope=0.76, dslope=0.01, intercept=-7.64, dintercept=0.02
    ):
        """Renzini & Peng (2015)

        https://ui.adsabs.harvard.edu/abs/2015ApJ...801L..29R/abstract
        """
        logM = np.log10(Mstar)
        logSFR = slope * logM + intercept
        sigma = np.sqrt((dslope * logM) ** 2 + dintercept**2)
        return logSFR, sigma


    logSFR, sigma = logSFR_with_errors(x_theory)
    sigma_3dex = 0.3  # added 0.3 dex according to paper
    SFR = 10**logSFR
    SFR_lo = 10 ** (logSFR - sigma_3dex)
    SFR_hi = 10 ** (logSFR + sigma_3dex)
    ax[1].fill_between(
        x_theory,
        SFR_lo,
        SFR_hi,
        facecolor="darkorange",
        alpha=0.4,
        zorder=0,
    )
    ax[1].plot(
        x_theory,
        SFR,
        ls="-.",
        color="darkorange",
        zorder=0,
        label=r"${\rm SFMS~ Renzini ~& ~Peng ~2015}$",
    )


    ## Saintonge and Catinella 2022: https://ui.adsabs.harvard.edu/abs/2022ARA%26A..60..319S/abstract
    def logSFR_MS_Saintonge_Catinella(Mstar):
        """
        from https://ui.adsabs.harvard.edu/abs/2022ARA%26A..60..319S/abstract
        this is derived from m Saintonge et al. (2017).
        Parameters
        ----------
        Mstar : array_like
            Stellar mass in solar masses (NOT logged)

        Returns
        -------
        logSFR : ndarray
            log10(SFR / Msun yr^-1)
        """
        M0 = 10**10.59
        alpha = -0.718
        logsfr = 0.412 - np.log10(1.0 + (Mstar / M0) ** alpha)
        return 10**logsfr


    ax[1].plot(
        x_theory,
        logSFR_MS_Saintonge_Catinella(x_theory),
        ls="--",
        color="k",
        zorder=0,
        label=r"${\rm SFMS~}$"
        + r"$~0.01 < z < 0.05$"
        + r"${~\rm Saintonge ~& ~Catinella ~2022}$",
    )

    # add 0.4 dex scatter region
    ax[1].fill_between(
        x_theory,
        logSFR_MS_Saintonge_Catinella(x_theory) * 10 ** (-0.4),
        logSFR_MS_Saintonge_Catinella(x_theory) * 10 ** (0.4),
        facecolor="grey",
        alpha=0.3,
        zorder=0,
    )
    # ax[1].fill_between(
    #     x_theory_extrapolation,
    #     logSFR_MS_Saintonge_Catinella(x_theory_extrapolation) * 10**(-0.4),
    #     logSFR_MS_Saintonge_Catinella(x_theory_extrapolation) * 10**(0.4),
    #     facecolor="tab:orange",
    #     alpha=0.3,
    #     zorder=0,
    # )


    # dwarf data from ManceraPina25_dwarfs.txt, Table G.2 of https://arxiv.org/pdf/2505.22727
    dwarf_data = np.loadtxt("./data/ManceraPina25_dwarfs.txt", skiprows=2)

    # all values are given in log
    logMstar = dwarf_data[:, 0]  # log(Mstar)
    logMstar_lo = dwarf_data[:, 1]  # lower sigma in log space
    logMstar_hi = dwarf_data[:, 2]  # upper sigma in log space

    logMgas = dwarf_data[:, 3]  # log(Mgas)
    logMgas_lo = dwarf_data[:, 4]  # lower sigma in log space
    logMgas_hi = dwarf_data[:, 5]  # upper sigma in log space


    Mstar = 10**logMstar
    Mgas = 10**logMgas


    Mstar_err_lo = Mstar - 10 ** (logMstar - logMstar_lo)
    Mstar_err_hi = 10 ** (logMstar + logMstar_hi) - Mstar

    Mgas_err_lo = Mgas - 10 ** (logMgas - logMgas_lo)
    Mgas_err_hi = 10 ** (logMgas + logMgas_hi) - Mgas

    fgas = Mgas / Mstar

    # --- propagate asymmetric errors in log space ---
    # log(fgas) = log(Mgas) - log(Mstar)
    fgas_log_lo = np.sqrt(logMgas_lo**2 + logMstar_hi**2)
    fgas_log_hi = np.sqrt(logMgas_hi**2 + logMstar_lo**2)

    fgas_err_lo = fgas * (1 - 10 ** (-fgas_log_lo))
    fgas_err_hi = fgas * (10 ** (fgas_log_hi) - 1)


    ax[0].errorbar(
        Mstar,
        fgas,
        xerr=[Mstar_err_lo, Mstar_err_hi],
        yerr=[fgas_err_lo, fgas_err_hi],
        fmt="o",
        color="brown",
        markersize=3,
        alpha=0.5,
        elinewidth=1,
        zorder=1,
    )

    ## ========================================================

    # make a custom legend for the observations
    # Create custom legend for observations
    custom_legend_elements = [
        Line2D(
            [0],
            [0],
            color="thistle",
            lw=2,
            label=r"$\rm{LTGs~double~power~law~Calette ~ et ~al. ~2018}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            markerfacecolor="k",
            markersize=3,
            markeredgecolor="k",
            linestyle="none",
            alpha=0.5,
            label=r"$\rm{xGASS}~$"
            + r"$0.01 < z < 0.05$"
            + r"$\rm{~Catinella ~et ~al. ~2018}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="brown",
            markerfacecolor="brown",
            markersize=3,
            markeredgecolor="brown",
            linestyle="none",
            alpha=0.5,
            label=r"$\rm{gas-rich~dwarfs}~$"
            + r"$z\sim0$"
            + r"$\rm{~ Mancera ~Piña ~et ~al. ~2025 }$",
        ),
    ]
    ax[0].legend(
        handles=custom_legend_elements, loc="upper right", frameon=False, fontsize=8
    )

    ax[1].legend(frameon=False, fontsize=8, loc="lower right")


    # ax[0].legend(
    #     ncols=4,
    #     frameon=False,
    #     bbox_to_anchor=(-0.05, 1.011),
    #     loc="lower left",
    #     title=r"redshift $z$",
    #     fontsize=10,
    # )

    ax[0].set(
        xscale="log", yscale="log", ylabel=r"$M_{\rm {ISM}}/M_\star$", ylim=(8e-2, 30)
    )
    ax[1].set(
        xscale="log",
        yscale="log",
        xlabel=r"$M_\star$ [M$_\odot$]",
        ylabel=r"${\rm SFR ~[M_\odot ~yr^{-1}]}$",
        xlim=(8e6, 8e11),
        ylim=(2e-4, 9),
    )


    # add text for the KS parameters used
    ax[0].set_title(
        rf"KS 1998: $\kappa={kappa_sfr:.4f}$, $n={n_sfr:.4f}$, $r_{{\rm disk}}={r_disk_sfr:.4f} R_{{\rm vir}}$",
        transform=ax[0].transAxes,
        fontsize=10,
    )

    # do a test run for a 1e12 Msun halo from 0.1 to 1 Gyr with these parameters and plot t_dep_eff = MISM/SFR to see 
    latest_model = CGMRegulator(
    1e12 * u.Msun,
    (0.1, 1),
    add_f_prevent_floor=1e-6,# virtually no floor
    KS_kappa_s = kappa_sfr,
    KS_n = n_sfr,
    disk_scale_length=r_disk_sfr,
    KS_parametrization="KS1998",
)
    latest_model.run_halo()
    results = latest_model.get_results()
    derived = latest_model.get_derived_quantities()
    # make an inset and plot t_dep_eff = MISM/SFR for this test run and cooling times 
    tcool = derived["tcool_real"]
    tdep_eff = results["m_star"] / derived["dot_m_sfr"]
    tff = derived["t_dynamical"]
    derived["f_prevent"]
    inset_ax_1 = ax[0].inset_axes([1.1, .33, 1 , 0.66])
    inset_ax_1.plot(derived["sim_time"], tdep_eff, lw=3, label=r"$t_{\rm dep,eff}$")
    # turn off x label
    inset_ax_1.set_xticks([])
    # plot tff as reference
    inset_ax_1.plot(derived["sim_time"], tff, color="k", linestyle="--", lw=3, label=r"$t_{\rm ff}$", alpha=0.5)
    
    inset_ax_2 = ax[1].inset_axes([1.1, 0.66, 1, 0.66])
    inset_ax_2.plot(derived["sim_time"], tcool, lw=3, label=r"$t_{\rm cool}$")
    inset_ax_2.plot(derived["sim_time"], tff, color="k", linestyle="--", lw=3, label=r"$t_{\rm ff}$", alpha=0.5)
    inset_ax_2.set_xticks([])
    
    inset_ax_3 = ax[1].inset_axes([1.1, 0, 1, 0.66])
    inset_ax_3.plot(derived["sim_time"], derived["f_prevent"], lw=3, label=r"$f_{\rm prevent}$")
    inset_ax_3.set_yscale("log")
    inset_ax_3.set_xlabel("time [Gyr]")
    
    inset_ax_1.set_yscale("log")
    inset_ax_2.set_yscale("log")
    inset_ax_1.set_title("test run for 1e12 Msun @z=0 halo")

    inset_ax_1.legend(frameon=False)
    inset_ax_2.legend(frameon=False)
    inset_ax_3.legend(frameon=False)
 
    # plt.savefig(
    #     f"./figures/KS98scan_nkappa_variation/{j:02d}_ism_gas_fractions_{param_txt}.png",
    #     dpi=200,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )
    plt.show()
# %% now, do single runs
mass_bins = 10
zobs = zbins_ctr
# make a unique halo array for each redshift
mhalos = np.geomspace(1e10, 1e13, mass_bins)
mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun
print(zobs)
print(mhalos[0])

#### XXX
n_sfr = 1.8
r_disk_sfr = 0.018
kappa_sfr = 0.02

param_txt = (
    f"KS_1998_kappa{str(kappa_sfr).replace('.', 'p')}_"
    + f"n{str(n_sfr).replace('.', 'p')}_"
    + f"r{str(r_disk_sfr).replace('.', 'p')}"
)
file = "./runs/smhm_2phase_redshift_scan_" f"{param_txt}.h5"

####

if not os.path.exists(file):
    print("running 2-phase model grid...", file)
    redshift_variation, zsims = run_2phase_model_redshift_grid(
        observe_at=zobs,  # redshift we want to observe
        mhalos=mhalos,
        write_to_file=file,
        disk_scale_length=r_disk_sfr,
        KS_n=n_sfr,
        KS_kappa_s=kappa_sfr,
        # add_f_prevent_floor=1e-6
    )


f = h5py.File(file, "r")
smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
mhalo_obs = f["Mhalo_obs"]
mstar = f["Mstar_obs"]
mism = f["MISM_obs"]
zs = f["redshifts"][:]
sfr = f["SFR_obs"]
mmetal_cgm = f["MMetals_obs"]
rvirs = virial_radius(zs, mhalo_obs * u.Msun)  # virial radius in kpc
halo_Tvirs = virial_T(mhalo_obs * u.Msun, rvirs)  # virial temperature in K
print(f.keys())
print(f["redshifts"][:])


fig, ax = plt.subplots(2, 1, figsize=(5, 6.5), dpi=300, sharex=True, sharey="row")
ax = ax.flatten()
plt.subplots_adjust(hspace=0.05)
# get a colormap from cmasher
cmap = plt.get_cmap("cmr.tropical")
colors = cmap(np.linspace(0, 1, len(zs)))
# replace the final color with grey for the lowest redshift
colors[-1] = [0.5, 0.5, 0.5, 1.0]
Tvir_max = 1e6 * u.K
# loop through redshift and plot mgas/mstar vs mstar
for i, z in enumerate(zs):
    Tvirs = halo_Tvirs[i]
    mask = Tvirs < Tvir_max

    ax[0].plot(
        mstar[i],
        mism[i] / mstar[i],
        color=colors[i],
        label=r"${:.1f}$".format(z),
        lw=3
    )
    ax[0].scatter(
        mstar[i][mask],
        mism[i][mask] / mstar[i][mask],
        s=30,
        color=colors[i],
        edgecolor="k",
        zorder=3,
    )

    t_depletion = mism[i] / sfr[i]
    sSFR_Gyr = sfr[i] / mstar[i]  # Gyr^-1
    sSFR_yr = sSFR_Gyr * 1e-9
    sfr_yr = sfr[i] * 1e-9
    ax[1].plot(
        mstar[i],
        sfr_yr,
        color=colors[i],
        lw=3
    )
    ax[1].scatter(
        mstar[i][mask], sfr_yr[mask], s=30, color=colors[i], edgecolor="k", zorder=3
    )

lines = ax[0].get_lines()
l1 = lines[0]
labelLine(
    l1,
    3e10,
    label=r"$z={}$".format(zs[0]),
    align=False,
    yoffset=0.0,
    ha="left",
    backgroundcolor="none",
    fontsize=7,
    # color="black",
)
labelLines(
    lines[1:],
    xvals=np.geomspace(1e9, 1e12, len(lines) - 1),
    yoffsets=0.01,
    align=False,
    fontsize=7,
    # ha="left",
    backgroundcolor="none",
    # color="black",
)


# make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right

# # Calculate the slope and intercept in log-log space
# x1, y1 = 8.7, 0.2
# x2, y2 = 10.35, -0.4

# # Generate x values spanning the plot range
# x_vals = np.linspace(6, 12, 100)
# # Compute corresponding y values for the line in log-log space
# slope = (y2 - y1) / (x2 - x1)
# y_vals = y1 + slope * (x_vals - x1)

# ax[0].plot(
#     10**x_vals,
#     10**y_vals,
#     ls="-",
#     color="k",
#     lw="4",
#     alpha=0.5,
#     zorder=1,
# )
# # put a text label near this line
# ax[0].text(
#     0.85,
#     0.22,
#     r"Calette+18, $z\sim 0$",
#     transform=ax[0].transAxes,
#     fontsize=8,
#     rotation=-35,
#     va="center",
#     ha="right",
# )

### ============== observations for comparison ==============

## xGass: Catinella et al: https://ui.adsabs.harvard.edu/abs/2018MNRAS.476..875C/abstract

# total Mgas/Mstar vs Mstar, table 2 average values per bin
catinella_mstar = 10 ** np.array([9.16, 9.44, 9.75, 10.05, 10.34, 10.65, 10.95, 11.21])
catinella_mgas_mstar = 10 ** np.array(
    [0.098, -0.136, -0.509, -0.518, -0.817, -0.958, -1.190, -1.328]
)
catinella_mgas_mstar_err = np.array(
    [0.064, 0.077, 0.076, 0.062, 0.055, 0.048, 0.048, 0.064]
)
yerr_lower = catinella_mgas_mstar * (1 - 10 ** (-catinella_mgas_mstar_err))
yerr_upper = catinella_mgas_mstar * (10 ** (catinella_mgas_mstar_err) - 1)
yerr = np.vstack([yerr_lower, yerr_upper])

# here for the raw counts, unbinnned unlike above, using the full table
# https://xgass.icrar.org/assets/data/xGASS_representative_sample.readme
# https://xgass.icrar.org/assets/data/xGASS_representative_sample.ascii
# 0.01 < z < 0.05 Catinella+18 still, but for some reason don't have full gas fraction so using binned values
xGass_dat = pd.read_csv(
    "./data/xGass.txt",
    delim_whitespace=True,
)
xgas_sfrbest = xGass_dat["SFR_best"]  # msun /year
xgas_mstar = 10 ** xGass_dat["lgMstar"]
detection_tag = xGass_dat["HIsrc"]  # 1 for detection, 0 for upper limit
alfalfa_tag = 1
mask = detection_tag == alfalfa_tag  # only the gas rich alfalfa detections
ax[0].errorbar(
    catinella_mstar,
    catinella_mgas_mstar,
    yerr=yerr,
    fmt="o",
    color="k",
    markersize=3,
    alpha=0.5,
    elinewidth=1,
)
ax[1].errorbar(
    xgas_mstar[mask],
    np.array(xgas_sfrbest[mask]),
    color="k",
    markersize=3,
    fmt="o",
    alpha=0.3,
    zorder=1,
)

# ssfr fit Steven Janowieck et al. 2020, xGASS: cold gas content and quenching in galaxies below the star-forming main sequence Eq https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.1982J/abstract, eq 1
# m_sfms = -0.344 # +/- 0.101
# b_sfrms = -9.822 # +/- 0.057
# m_sfs = 0.088 # +/- 0.028
# b_sfs = 0.188 # +/- 0.036
# m_star_test = np.log10(np.geomspace(1e9, 1e11, 100))
# log_sFR_ms = m_sfms * (m_star_test - 9) + b_sfrms
# # now translate to sSFR for plotting in panel 1
# sSFR_ms = 10**log_sFR_ms
# ax[1].plot(10**m_star_test, sSFR_ms, ls='--', color='k', label='xGASS SFMS (Janowiecki+20)')


## https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.1526P/abstract Popesso eneded up no using


## Callette relation https://arxiv.org/pdf/1803.07692, late type galaxy Rgas-Mstar
log_C_ltg = 4.76
log_C_err = 0.05
a_ltg = -0.52
a_err = 0.03
intrinsic_scatter_dex = 0.44

### early type galaxies
log_C_etg = 3.70
a_etg = -0.58
intrinsic_scatter_dex_etg = 0.68

# Plot Calette relation: y(M*) = C * (M*/M_sun)^a
log_m_star_thry = np.linspace(6, 12, 100)
m_star_thry = 10**log_m_star_thry

# y_catinella_etg = 10**log_C_etg * (m_star_thry)**a_etg
# ax[0].plot(m_star_thry, y_catinella_etg, ls='--', color='k', lw=2, label='Callette+18 ETG', zorder=2)

# now the late type galaxies
y_catinella_ltg = 10**log_C_ltg * (m_star_thry) ** a_ltg
scatter_ltg = 0.44

y_upper = y_catinella_ltg * 10**scatter_ltg
y_lower = y_catinella_ltg / 10**scatter_ltg
# ax[0].plot(
#     m_star_thry,
#     y_catinella_ltg,
#     ls="-",
#     color="k",
#     lw=2,
#     label="Callette+18 LTG",
#     zorder=0,
# )
# ax[0].fill_between(m_star_thry, y_lower, y_upper, facecolor='k', alpha=0.2, zorder=0)

# Calette but double power law, still LTG
C = 1.69
a = 0.18
b = 0.61
log_Mtrunc = 9.2
intrinsic_scatter_dbl_pl = 0.44
y_dbl_power = C / (
    (m_star_thry / 10**log_Mtrunc) ** (a) + (m_star_thry / 10**log_Mtrunc) ** b
)
ax[0].plot(
    m_star_thry,
    y_dbl_power,
    ls="-.",
    color="thistle",
    lw=2,
    label="Calette+18 LTG DPL",
    zorder=0,
)
y_dbl_power_upper = y_dbl_power * 10**intrinsic_scatter_dbl_pl
y_dbl_power_lower = y_dbl_power / 10**intrinsic_scatter_dbl_pl
ax[0].fill_between(
    m_star_thry,
    y_dbl_power_lower,
    y_dbl_power_upper,
    facecolor="thistle",
    alpha=0.5,
    zorder=0,
)


##### done by hand
# # Calculate the slope and intercept in log-log space
# x1, y1 = 8.7, 0.2
# x2, y2 = 10.35, -0.4

# # Generate x values spanning the plot range
# x_vals = np.linspace(6, 12, 100)
# # Compute corresponding y values for the line in log-log space
# slope = (y2 - y1) / (x2 - x1)
# y_vals = y1 + slope * (x_vals - x1)

# ax[0].plot(
#     10**x_vals,
#     10**y_vals,
#     ls="-",
#     color="k",
#     lw="4",
#     alpha=0.5,
#     zorder=1,
# )

## simulations
simba_cat = h5py.File("./data/m50n512_153.hdf5", "r")  # no AGN
# simba_gals = simba_cat["galaxy_data"]
halo_HI_gas = simba_cat["halo_data/dicts/masses.HI"][:]
halo_H2_gas = simba_cat["halo_data/dicts/masses.H2"][:]
halo_gas = simba_cat["halo_data/dicts/masses.gas"][:]
halo_total = simba_cat["halo_data/dicts/masses.total"][:]
halo_stellar_mass = simba_cat["halo_data/dicts/masses.stellar"][:]
halo_sfr = simba_cat["/halo_data/sfr"][:]
halo_gas_mass_fraction = halo_gas / halo_total
# ax[0].scatter(
#     halo_stellar_mass,
#     halo_gas / halo_stellar_mass,
#     s=0.1,
#     alpha=0.5,
#     color="gray",
#     label="SIMBA No-AGN z=0, MH2/ Mstar (Dave+19)",
# )
# ax[1].scatter(
#     halo_stellar_mass,
#     halo_sfr ,
#      s=0.1,
#     alpha=0.5,
#     color="gray")


x_theory = np.geomspace(1e6, 1e12, 20)  # some theory lines


# Renzini & Peng 2015 SFMS with 0.3 dex scatter
def logSFR_with_errors(
    Mstar, slope=0.76, dslope=0.01, intercept=-7.64, dintercept=0.02
):
    """Renzini & Peng (2015)

    https://ui.adsabs.harvard.edu/abs/2015ApJ...801L..29R/abstract
    """
    logM = np.log10(Mstar)
    logSFR = slope * logM + intercept
    sigma = np.sqrt((dslope * logM) ** 2 + dintercept**2)
    return logSFR, sigma


logSFR, sigma = logSFR_with_errors(x_theory)
sigma_3dex = 0.3  # added 0.3 dex according to paper
SFR = 10**logSFR
SFR_lo = 10 ** (logSFR - sigma_3dex)
SFR_hi = 10 ** (logSFR + sigma_3dex)
ax[1].fill_between(
    x_theory,
    SFR_lo,
    SFR_hi,
    facecolor="darkorange",
    alpha=0.4,
    zorder=0,
)
ax[1].plot(
    x_theory,
    SFR,
    ls="-.",
    color="darkorange",
    zorder=0,
    label=r"${\rm SFMS~ (Renzini ~& ~Peng ~2015)}$",
)


## Saintonge and Catinella 2022: https://ui.adsabs.harvard.edu/abs/2022ARA%26A..60..319S/abstract
def logSFR_MS_Saintonge_Catinella(Mstar):
    """
    from https://ui.adsabs.harvard.edu/abs/2022ARA%26A..60..319S/abstract
    this is derived from m Saintonge et al. (2017).
    Parameters
    ----------
    Mstar : array_like
        Stellar mass in solar masses (NOT logged)

    Returns
    -------
    logSFR : ndarray
        log10(SFR / Msun yr^-1)
    """
    M0 = 10**10.59
    alpha = -0.718
    logsfr = 0.412 - np.log10(1.0 + (Mstar / M0) ** alpha)
    return 10**logsfr


ax[1].plot(
    x_theory,
    logSFR_MS_Saintonge_Catinella(x_theory),
    ls="--",
    color="k",
    zorder=0,
    label=r"${\rm SFMS~}$"
    + r"$~0.01 < z < 0.05$"
    + r"${~\rm (Saintonge ~& ~Catinella ~2022)}$",
)

# add 0.4 dex scatter region
ax[1].fill_between(
    x_theory,
    logSFR_MS_Saintonge_Catinella(x_theory) * 10 ** (-0.4),
    logSFR_MS_Saintonge_Catinella(x_theory) * 10 ** (0.4),
    facecolor="grey",
    alpha=0.3,
    zorder=0,
)
# ax[1].fill_between(
#     x_theory_extrapolation,
#     logSFR_MS_Saintonge_Catinella(x_theory_extrapolation) * 10**(-0.4),
#     logSFR_MS_Saintonge_Catinella(x_theory_extrapolation) * 10**(0.4),
#     facecolor="tab:orange",
#     alpha=0.3,
#     zorder=0,
# )


# dwarf data from ManceraPina25_dwarfs.txt, Table G.2 of https://arxiv.org/pdf/2505.22727
dwarf_data = np.loadtxt("./data/ManceraPina25_dwarfs.txt", skiprows=2)

# all values are given in log
logMstar = dwarf_data[:, 0]  # log(Mstar)
logMstar_lo = dwarf_data[:, 1]  # lower sigma in log space
logMstar_hi = dwarf_data[:, 2]  # upper sigma in log space

logMgas = dwarf_data[:, 3]  # log(Mgas)
logMgas_lo = dwarf_data[:, 4]  # lower sigma in log space
logMgas_hi = dwarf_data[:, 5]  # upper sigma in log space


Mstar = 10**logMstar
Mgas = 10**logMgas


Mstar_err_lo = Mstar - 10 ** (logMstar - logMstar_lo)
Mstar_err_hi = 10 ** (logMstar + logMstar_hi) - Mstar

Mgas_err_lo = Mgas - 10 ** (logMgas - logMgas_lo)
Mgas_err_hi = 10 ** (logMgas + logMgas_hi) - Mgas

fgas = Mgas / Mstar

# --- propagate asymmetric errors in log space ---
# log(fgas) = log(Mgas) - log(Mstar)
fgas_log_lo = np.sqrt(logMgas_lo**2 + logMstar_hi**2)
fgas_log_hi = np.sqrt(logMgas_hi**2 + logMstar_lo**2)

fgas_err_lo = fgas * (1 - 10 ** (-fgas_log_lo))
fgas_err_hi = fgas * (10 ** (fgas_log_hi) - 1)


ax[0].errorbar(
    Mstar,
    fgas,
    xerr=[Mstar_err_lo, Mstar_err_hi],
    yerr=[fgas_err_lo, fgas_err_hi],
    fmt="o",
    color="brown",
    markersize=3,
    alpha=0.5,
    elinewidth=1,
    zorder=1,
)

## ========================================================

# make a custom legend for the observations
# Create custom legend for observations
custom_legend_elements = [
    Line2D(
        [0],
        [0],
        color="thistle",
        lw=2,
        label=r"$\rm{LTGs~double~power~law~(Calette ~ et ~al. ~2018)}$",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="k",
        markerfacecolor="k",
        markersize=3,
        markeredgecolor="k",
        linestyle="none",
        alpha=0.5,
        label=r"$\rm{xGASS}~$"
        + r"$0.01 < z < 0.05$"
        + r"$\rm{~(Catinella ~et ~al. ~2018)}$",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="brown",
        markerfacecolor="brown",
        markersize=3,
        markeredgecolor="brown",
        linestyle="none",
        alpha=0.5,
        label=r"$\rm{gas-rich~dwarfs}~$"
        + r"$z\sim0$"
        + r"$\rm{~ (Mancera ~Piña ~et ~al. ~2025)}$",
    ),
]
ax[0].legend(
    handles=custom_legend_elements, loc="upper right", frameon=False, fontsize=8.5
)

ax[1].legend(frameon=False, fontsize=9, loc="lower right")


# ax[0].legend(
#     ncols=4,
#     frameon=False,
#     bbox_to_anchor=(-0.05, 1.011),
#     loc="lower left",
#     title=r"redshift $z$",
#     fontsize=10,
# )

ax[0].set(
    xscale="log", yscale="log", ylabel=r"$M_{\rm {ISM}}/M_\star$", ylim=(0.15, 30)
)
ax[1].set(
    xscale="log",
    yscale="log",
    xlabel=r"$\rm M_\star [M_\odot]$",
    ylabel=r"${\rm SFR ~[M_\odot ~yr^{-1}]}$",
    xlim=(8e6, 8e11),
    ylim=(2e-4, 9),
)


# add text for the KS parameters used
# ax[0].set_title(
#     rf"KS 1998: $\kappa={kappa_sfr:.4f}$, $n={n_sfr:.4f}$, $r_{{\rm disk}}={r_disk_sfr:.4f} R_{{\rm vir}}$",
#     transform=ax[0].transAxes,
#     fontsize=10,
# )

plt.savefig(
    f"./figures/ism_gas_fractions_{param_txt}.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.show()
# %% metallicity plot

z_sun = 0.0134
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), dpi=300)
for i, z in enumerate(zs):
    metallicity = (mmetal_cgm[i] / mism[i]) / z_sun
    ax2.plot(
        mhalo_obs[i],
        metallicity,
        label=f"{z:.1f}",
        marker="o",
        color=colors[i],
        markeredgecolor="k",
    )

ax2.set(
    xscale="log",
    yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    ylabel=r"CGM Metallicity [$Z_\odot$]",
)
ax2.legend(
    ncols=4,
    frameon=False,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title=r"redshift $z$",
    fontsize=10,
)
plt.savefig(
    "./figures/cgm_metallicity_2phase.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

# %%

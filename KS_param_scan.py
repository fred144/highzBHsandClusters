# %% scan through the KS_n and KS_kappa_s of the KS law for the 2 phase model

import numpy as np
import matplotlib.pyplot as plt
from cgm_sf_regulator import CGMRegulator
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u

#

mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 13)  # gyrs
n_samples = 10
n_fixed = 1.5  # keep n fixed for now
kappa_s_fixed = 1.0
disk_scale_length_fixed = 0.02  # of rvir
KS_parametrization_fixed = "KS1998"
TEST_tej_Tvir_definition = True
# %% Store results for each kappa_s
timescales = {
    "tcool_real": [],
    "t_dynamical": [],
    "t_dep_effect": [],
    "t_ejection": [],
    "sim_time": [],
}
sf_statistics = {
    "SFR": [],
    "SFE": [],
    "m_star": [],
    "m_ism": [],
    "m_cgm_hot": [],
    "m_cgm_cold": [],
    "egy_cgm": [],
    "egy_ism_wind": [],
    "egy_radloss": [],
    "egy_eject": [],
    "egy_accrete": [],
    "dot_m_cgm_in": [],
    "dot_m_cgm_out": [],
    "f_prevent": [],
}

kappa_s_to_try = np.logspace(-1, 1, n_samples)
for kappa_s in kappa_s_to_try:
    model = CGMRegulator(
        mhalo_z0,
        t_span,
        KS_kappa_s=kappa_s,
        KS_n=n_fixed,
        disk_scale_length=disk_scale_length_fixed,
        add_f_prevent_floor=1e-6,
        KS_parametrization=KS_parametrization_fixed,
        TEST_tej_Tvir_definition=TEST_tej_Tvir_definition,
    )
    model.run_halo()
    results = model.get_results()
    derived = model.get_derived_quantities()
    timescales["tcool_real"].append(derived["tcool_real"])
    timescales["t_dynamical"].append(derived["t_dynamical"])
    timescales["t_dep_effect"].append(results["m_star"] / derived["dot_m_sfr"])
    timescales["t_ejection"].append(derived["t_ejection"])
    timescales["sim_time"].append(derived["sim_time"])

    sf_statistics["SFR"].append(derived["dot_m_sfr"])
    sf_statistics["SFE"].append(derived["halo_sfe"])
    sf_statistics["m_star"].append(results["m_star"])
    sf_statistics["m_ism"].append(results["m_ism"])

    sf_statistics["m_cgm_hot"].append(results["m_cgm_hot"])
    sf_statistics["m_cgm_cold"].append(results["m_cgm_cold"])
    sf_statistics["egy_cgm"].append(results["egy_cgm"])
    sf_statistics["egy_ism_wind"].append(results["egy_ism_wind"])
    sf_statistics["egy_radloss"].append(results["egy_radloss"])
    sf_statistics["egy_eject"].append(results["egy_eject"])
    sf_statistics["egy_accrete"].append(results["egy_accrete"])
    
    sf_statistics["dot_m_cgm_in"].append(derived["dot_m_cgm_in"])
    sf_statistics["dot_m_cgm_out"].append(derived["dot_m_cgm_out"])
    sf_statistics["f_prevent"].append(derived["f_prevent"])


# %% Scan for KS_n values
n_to_try = np.linspace(1.0, 2.0, n_samples)
timescales_n = {
    "tcool_real": [],
    "t_dynamical": [],
    "t_dep_effect": [],
    "t_ejection": [],
    "sim_time": [],
}
sf_statistics_n = {
    "SFR": [],
    "SFE": [],
    "m_star": [],
    "m_ism": [],
    "m_cgm_hot": [],
    "m_cgm_cold": [],
    "egy_cgm": [],
    "egy_ism_wind": [],
    "egy_radloss": [],
    "egy_eject": [],
    "egy_accrete": [],
     "dot_m_cgm_in": [],
    "dot_m_cgm_out": [],
    "f_prevent": [],
}

for n_val in n_to_try:
    model_n = CGMRegulator(
        mhalo_z0,
        t_span,
        KS_kappa_s=kappa_s_fixed,
        KS_n=n_val,
        disk_scale_length=disk_scale_length_fixed,
        add_f_prevent_floor=1e-6,
        KS_parametrization=KS_parametrization_fixed,
        TEST_tej_Tvir_definition=TEST_tej_Tvir_definition,
    )
    model_n.run_halo()
    results_n = model_n.get_results()
    derived_n = model_n.get_derived_quantities()
    timescales_n["tcool_real"].append(derived_n["tcool_real"])
    timescales_n["t_dynamical"].append(derived_n["t_dynamical"])
    timescales_n["t_dep_effect"].append(results_n["m_star"] / derived_n["dot_m_sfr"])
    timescales_n["t_ejection"].append(derived_n["t_ejection"])
    timescales_n["sim_time"].append(derived_n["sim_time"])
    sf_statistics_n["SFR"].append(derived_n["dot_m_sfr"])
    sf_statistics_n["SFE"].append(derived_n["halo_sfe"])
    sf_statistics_n["m_star"].append(results_n["m_star"])
    sf_statistics_n["m_ism"].append(results_n["m_ism"])

    sf_statistics_n["m_cgm_hot"].append(results_n["m_cgm_hot"])
    sf_statistics_n["m_cgm_cold"].append(results_n["m_cgm_cold"])
    sf_statistics_n["egy_cgm"].append(results_n["egy_cgm"])
    sf_statistics_n["egy_ism_wind"].append(results_n["egy_ism_wind"])
    sf_statistics_n["egy_radloss"].append(results_n["egy_radloss"])
    sf_statistics_n["egy_eject"].append(results_n["egy_eject"])
    sf_statistics_n["egy_accrete"].append(results_n["egy_accrete"])
    
    sf_statistics_n["dot_m_cgm_in"].append(derived_n["dot_m_cgm_in"])
    sf_statistics_n["dot_m_cgm_out"].append(derived_n["dot_m_cgm_out"])
    sf_statistics_n["f_prevent"].append(derived_n["f_prevent"])

# %% now for disk
r_disk_to_try = np.linspace(0.01, 0.1, n_samples)
# r_disk_to_try = np.sort(np.append(r_disk_to_try, disk_scale_length_fixed))
timescales_disk = {
    "tcool_real": [],
    "t_dynamical": [],
    "t_dep_effect": [],
    "t_ejection": [],
    "sim_time": [],
}
sf_statistics_disk = {
    "SFR": [],
    "SFE": [],
    "m_star": [],
    "m_ism": [],
    "m_cgm_hot": [],
    "m_cgm_cold": [],
    "egy_cgm": [],
    "egy_ism_wind": [],
    "egy_radloss": [],
    "egy_eject": [],
    "egy_accrete": [],
     "dot_m_cgm_in": [],
    "dot_m_cgm_out": [],
    "f_prevent": [],
}

for r_disk in r_disk_to_try:
    model_disk = CGMRegulator(
        mhalo_z0,
        t_span,
        KS_kappa_s=kappa_s_fixed,
        KS_n=n_fixed,
        disk_scale_length=r_disk,
        add_f_prevent_floor=1e-6,
        KS_parametrization=KS_parametrization_fixed,
        TEST_tej_Tvir_definition=TEST_tej_Tvir_definition,
        
    )
    model_disk.run_halo()
    results_disk = model_disk.get_results()
    derived_disk = model_disk.get_derived_quantities()
    timescales_disk["tcool_real"].append(derived_disk["tcool_real"])
    timescales_disk["t_dynamical"].append(derived_disk["t_dynamical"])
    timescales_disk["t_dep_effect"].append(
        results_disk["m_star"] / derived_disk["dot_m_sfr"]
    )
    timescales_disk["t_ejection"].append(derived_disk["t_ejection"])
    timescales_disk["sim_time"].append(derived_disk["sim_time"])
    sf_statistics_disk["SFR"].append(derived_disk["dot_m_sfr"])
    sf_statistics_disk["SFE"].append(derived_disk["halo_sfe"])
    sf_statistics_disk["m_star"].append(results_disk["m_star"])
    sf_statistics_disk["m_ism"].append(results_disk["m_ism"])
    sf_statistics_disk["m_cgm_hot"].append(results_disk["m_cgm_hot"])
    sf_statistics_disk["m_cgm_cold"].append(results_disk["m_cgm_cold"])
    sf_statistics_disk["egy_cgm"].append(results_disk["egy_cgm"])
    sf_statistics_disk["egy_ism_wind"].append(results_disk["egy_ism_wind"])
    sf_statistics_disk["egy_radloss"].append(results_disk["egy_radloss"])
    sf_statistics_disk["egy_eject"].append(results_disk["egy_eject"])
    sf_statistics_disk["egy_accrete"].append(results_disk["egy_accrete"])
    
    
    # 
    sf_statistics_disk["dot_m_cgm_in"].append(derived_disk["dot_m_cgm_in"])
    sf_statistics_disk["dot_m_cgm_out"].append(derived_disk["dot_m_cgm_out"])
    sf_statistics_disk["f_prevent"].append(derived_disk["f_prevent"])

# %%
# Plotting
fig, ax = plt.subplots(3, 3, figsize=(10, 8), sharex=True, dpi=300, sharey="row")
plt.subplots_adjust(hspace=0, wspace=0)

# --- Left column: kappa_s scan ---
cmap_kappa = plt.get_cmap("coolwarm")
norm_kappa = mpl.colors.LogNorm(vmin=kappa_s_to_try.min(), vmax=kappa_s_to_try.max())
sm_kappa = mpl.cm.ScalarMappable(norm=norm_kappa, cmap=cmap_kappa)

for i, kappa_s in enumerate(kappa_s_to_try):
    color = cmap_kappa(norm_kappa(kappa_s))
    ax[0, 0].plot(timescales["sim_time"][i], timescales["tcool_real"][i], color=color)
    ax[1, 0].plot(timescales["sim_time"][i], timescales["t_dep_effect"][i], color=color)
    ax[2, 0].plot(timescales["sim_time"][i], timescales["t_ejection"][i], color=color)

ax[0, 0].set_ylabel(r"$t_{\mathrm{cool}}$ [Gyr]")
ax[1, 0].set_ylabel(r"$t_{\mathrm{dep, eff}}$ [Gyr]")
ax[2, 0].set_ylabel(r"$t_{\mathrm{eject}}$ [Gyr]")
ax[2, 0].set_xlabel("Time [Gyr]")

# Add colorbar for kappa_s
cax_kappa = ax[0, 0].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax[0, 0].transAxes)
cb_kappa = fig.colorbar(sm_kappa, cax=cax_kappa, orientation="horizontal")
cax_kappa.set_title(r"$\kappa_s = {} {{\rm ~is~usually~fixed~to~this}}$".format(kappa_s_fixed))
cax_kappa.xaxis.set_ticks_position("top")

# --- Middle column: n scan ---
cmap_n = plt.get_cmap("coolwarm")
norm_n = mpl.colors.Normalize(vmin=n_to_try.min(), vmax=n_to_try.max())
sm_n = mpl.cm.ScalarMappable(norm=norm_n, cmap=cmap_n)

for i, n_val in enumerate(n_to_try):
    color = cmap_n(norm_n(n_val))
    ax[0, 1].plot(
        timescales_n["sim_time"][i], timescales_n["tcool_real"][i], color=color
    )
    ax[1, 1].plot(
        timescales_n["sim_time"][i], timescales_n["t_dep_effect"][i], color=color
    )
    ax[2, 1].plot(
        timescales_n["sim_time"][i], timescales_n["t_ejection"][i], color=color
    )


ax[2, 1].set_xlabel("Time [Gyr]")

# Add colorbar for n
cax_n = ax[0, 1].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax[0, 1].transAxes)
cb_n = fig.colorbar(sm_n, cax=cax_n, orientation="horizontal")
cax_n.set_title(r"$n = {} {{\rm ~is~usually~fixed~to~this}}$".format(n_fixed))
cax_n.xaxis.set_ticks_position("top")

# --- Right column: r_disk scan ---
cmap_disk = plt.get_cmap("coolwarm")
norm_disk = mpl.colors.Normalize(vmin=r_disk_to_try.min(), vmax=r_disk_to_try.max())
sm_disk = mpl.cm.ScalarMappable(norm=norm_disk, cmap=cmap_disk)

for i, r_disk in enumerate(r_disk_to_try):
    color = cmap_disk(norm_disk(r_disk))
    ax[0, 2].plot(
        timescales_disk["sim_time"][i], timescales_disk["tcool_real"][i], color=color
    )
    ax[1, 2].plot(
        timescales_disk["sim_time"][i], timescales_disk["t_dep_effect"][i], color=color
    )
    ax[2, 2].plot(
        timescales_disk["sim_time"][i], timescales_disk["t_ejection"][i], color=color
    )


ax[2, 2].set_xlabel("Time [Gyr]")

# Add colorbar for r_disk
cax_disk = ax[0, 2].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax[0, 2].transAxes)
cb_disk = fig.colorbar(sm_disk, cax=cax_disk, orientation="horizontal")
cax_disk.set_title(
    r"$r_{{\rm disk}}/r_{{\rm vir}} = {}  {{\rm ~is~usually~fixed~to~this}}$".format(disk_scale_length_fixed)
)
cax_disk.xaxis.set_ticks_position("top")

# Common formatting and free fall time reference
for i, a in enumerate(ax.flatten()):
    a.set(yscale="log", ylim=(1e-3, 2), xlim=(0.15, 0.5))
    label = "free fall time"
    if i == 0:
        a.plot(
            timescales_n["sim_time"][0],
            timescales_n["t_dynamical"][0],
            color="grey",
            linestyle="--",
            lw=2,
            label=label,
        )
    else:
        a.plot(
            timescales_n["sim_time"][0],
            timescales_n["t_dynamical"][0],
            color="grey",
            linestyle="--",
            lw=2,
        )

ax[0, 0].legend(frameon=False)
plt.suptitle(KS_parametrization_fixed)
plt.savefig(
    "./figures/KS_param_scan_{}.png".format(KS_parametrization_fixed), dpi=300, bbox_inches="tight", pad_inches=0.05
)

plt.show()


# %%
# Plot SFE, SFR, m_gas, m_star, m_cgm_hot, m_cgm_cold, egy cgm, f_prevent for each parameter scan
fig2, ax2 = plt.subplots(8, 3, figsize=(10, 14), sharex=True, dpi=300, sharey="row")
plt.subplots_adjust(hspace=0.0, wspace=0.0)

# --- Left column: kappa_s scan ---
for i, kappa_s in enumerate(kappa_s_to_try):
    color = cmap_kappa(norm_kappa(kappa_s))
    ax2[0, 0].plot(timescales["sim_time"][i], sf_statistics["SFE"][i], color=color)
    ax2[1, 0].plot(timescales["sim_time"][i], sf_statistics["SFR"][i], color=color)
    ax2[2, 0].plot(timescales["sim_time"][i], sf_statistics["m_ism"][i], color=color)
    ax2[3, 0].plot(timescales["sim_time"][i], sf_statistics["m_star"][i], color=color)
    ax2[4, 0].plot(
        timescales["sim_time"][i], sf_statistics["m_cgm_hot"][i], color=color
    )
    ax2[5, 0].plot(
        timescales["sim_time"][i], sf_statistics["m_cgm_cold"][i], color=color
    )
    ax2[6, 0].plot(timescales["sim_time"][i], sf_statistics["egy_cgm"][i], color=color)
    ax2[7, 0].plot(timescales["sim_time"][i], sf_statistics["f_prevent"][i], color=color)

ax2[0, 0].set_ylabel("SFE")
ax2[0, 0].set_ylim(2e-3, None)
ax2[1, 0].set_ylabel("SFR [Msun/Gyr]")
ax2[1, 0].set_ylim(1e7, None)
ax2[2, 0].set_ylabel(r"$M_{\mathrm{ISM}}$ [Msun]")
ax2[2, 0].set_ylim(5e5, 5e9)
ax2[3, 0].set_ylabel(r"$M_{\star}$ [Msun]")
ax2[3, 0].set_ylim(5e5, 5e9)
ax2[4, 0].set_ylabel(r"$M_{\mathrm{CGM, hot}}$ [Msun]")
ax2[4, 0].set_ylim(5e5, 1e10)
ax2[5, 0].set_ylabel(r"$M_{\mathrm{CGM, cold}}$ [Msun]")
ax2[5, 0].set_ylim(5e5, 1e10)
ax2[6, 0].set_ylabel(r"$E_{\mathrm{CGM}}$ [erg]")
ax2[6, 0].set_ylim(1e53, None)
ax2[7, 0].set_ylabel(r"$f_{\mathrm{prevent}}$")
ax2[7, 0].set_xlabel("Time [Gyr]")

cax2_kappa = ax2[0, 0].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax2[0, 0].transAxes)
fig2.colorbar(sm_kappa, cax=cax2_kappa, orientation="horizontal")
cax2_kappa.set_title(r"$\kappa_s = {} {{\rm is usually fixed}}$".format(kappa_s_fixed))
cax2_kappa.xaxis.set_ticks_position("top")

# --- Middle column: n scan ---
for i, n_val in enumerate(n_to_try):
    color = cmap_n(norm_n(n_val))
    ax2[0, 1].plot(timescales_n["sim_time"][i], sf_statistics_n["SFE"][i], color=color)
    ax2[1, 1].plot(timescales_n["sim_time"][i], sf_statistics_n["SFR"][i], color=color)
    ax2[2, 1].plot(
        timescales_n["sim_time"][i], sf_statistics_n["m_ism"][i], color=color
    )
    ax2[3, 1].plot(
        timescales_n["sim_time"][i], sf_statistics_n["m_star"][i], color=color
    )
    ax2[4, 1].plot(
        timescales_n["sim_time"][i], sf_statistics_n["m_cgm_hot"][i], color=color
    )
    ax2[5, 1].plot(
        timescales_n["sim_time"][i], sf_statistics_n["m_cgm_cold"][i], color=color
    )
    ax2[6, 1].plot(
        timescales_n["sim_time"][i], sf_statistics_n["egy_cgm"][i], color=color
    )
    ax2[7, 1].plot(
        timescales_n["sim_time"][i], sf_statistics_n["f_prevent"][i], color=color
    )

ax2[7, 1].set_xlabel("Time [Gyr]")

cax2_n = ax2[0, 1].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax2[0, 1].transAxes)
fig2.colorbar(sm_n, cax=cax2_n, orientation="horizontal")
cax2_n.set_title(r"$n = {} {{\rm is usually fixed}}$".format(n_fixed))
cax2_n.xaxis.set_ticks_position("top")

# --- Right column: r_disk scan ---
for i, r_disk in enumerate(r_disk_to_try):
    color = cmap_disk(norm_disk(r_disk))
    ax2[0, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["SFE"][i], color=color
    )
    ax2[1, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["SFR"][i], color=color
    )
    ax2[2, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["m_ism"][i], color=color
    )
    ax2[3, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["m_star"][i], color=color
    )
    ax2[4, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["m_cgm_hot"][i], color=color
    )
    ax2[5, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["m_cgm_cold"][i], color=color
    )
    ax2[6, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["egy_cgm"][i], color=color
    )
    ax2[7, 2].plot(
        timescales_disk["sim_time"][i], sf_statistics_disk["f_prevent"][i], color=color
    )

ax2[7, 2].set_xlabel("Time [Gyr]")

cax2_disk = ax2[0, 2].inset_axes([0.05, 1.1, 0.9, 0.05], transform=ax2[0, 2].transAxes)
fig2.colorbar(sm_disk, cax=cax2_disk, orientation="horizontal")
cax2_disk.set_title(
    r"$r_{{\rm disk}}/r_{{\rm vir}} = {}  {{\rm ~is~usually~fixed~to~this}}$".format(disk_scale_length_fixed)
)
cax2_disk.xaxis.set_ticks_position("top")

# Formatting
for i in range(8):
    for j in range(3):
        ax2[i, j].set(xlim=(t_span[0], 1), yscale="log")

plt.suptitle(KS_parametrization_fixed, y=0.95)

plt.savefig(
    "./figures/KS_param_scan_SFE_SFR_mgas_mstar_fprev_{}.png".format(KS_parametrization_fixed),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
# %%

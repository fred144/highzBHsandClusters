#%%
from cgm_sf_regulator import CGM_regulator, plot_halo_diagnostics
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as consts
import numpy as np

mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 12.5)  # gyrs
eta_m = 0.1
eta_e = 0.1
eta_z = 0.2


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo

    """
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)
circular_velocity
#%%
model = CGM_regulator(

    mhalo_z0, t_span, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z, cooling_dynamic_time_norm=1
)
run = model.run_halo()
results = model.get_results()
derived = model.get_derived_quantities()
halo_masses = results["m_halo"]
halo_rvir = derived["halo_rvir"]

sfr = derived["dot_m_sfr"]
z = derived["z"]
t = derived["sim_time"]
halo_sfe = derived["f_star"][-1]
# plot_halo_profile(results, derived)
plot_halo_diagnostics(
    results,
    derived,
    title="new energy loading 0.1 * (halo_vcirc / 200) ** (-3/2), old CGM profile, full dynamical added to cooling time",
)

#%%

# ism mass, depletion time 
m_ism = results["m_gas"]
t_dep = derived["t_dep"]

def surf_dens_sfr(mism, rvir, Asfr, n = 1.5):
    """
    Calculate the surface density of star formation rate (SFR) as a function of radius.
    
    Parameters
    ----------
    mism : float
        The mass of the ISM

    """
    # assuming an exponential profile for the ISM
    r_trunc = 0.02 * rvir
    sigma0 = mism / (2 * np.pi * r_trunc**2) # msun / kpc^2
    
    kappa_s = 1
    Asfr = 1e-12 * kappa_s * 1e9 # msun / yr / kpc^2
    dot_m_star = Asfr * sigma0**n * (2 * np.pi *r_trunc**2) / n**2
    
    ## resolved KS law from paper
    
    classic_dot_m_star = (1e-12 * kappa_s *  sigma0**n) * r_trunc**2 #msun /yr
    
    return dot_m_star, sigma0,    classic_dot_m_star * 1e9
    
# let's plot the sfr vs time
fig, ax =  plt.subplots(3,1, figsize=(5, 7), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.0)
ax[0].plot(t, sfr, label="SFR")
ax[0].set(xlabel="Time [Gyr]", ylabel="SFR [Msun/Gyr]", yscale="log")
ax0_inset = ax[0].inset_axes([0.5, 0.5, 0.4, 0.4])  # [x, y, width, height]
ax0_inset.plot(t, m_ism, label="ISM mass")
ax0_inset.set(xlabel="Time [Gyr]", ylabel="ism \n mass  [Msun]", yscale="log")
ax0_inset1 = ax[0].inset_axes([0.5, 0.1, 0.4, 0.4])  # [x, y, width, height]
ax0_inset1.plot(t, t_dep, label="Halo mass")
ax0_inset1.set(xlabel="Time [Gyr]", ylabel="depletion \n time [Gyr] ", yscale="log")

# calculate the surface density based sfr
Asfr = 0.01
n = 1.5
dot_m_star, sigma0, classic_dot_m_star = surf_dens_sfr(m_ism, halo_rvir, Asfr, n )

ax[1].plot(t, dot_m_star, label="SFR, A={}, n={}".format(Asfr, n))
ax[1].plot(t, classic_dot_m_star, label=r"$\Sigma_{{\rm SFR}} = \kappa_s 10^{{-12}} \Sigma^{n}_{{0}}$")
ax[1].set(xlabel="Time [Gyr]", ylabel="SFR [Msun/Gyr]", yscale="log")
ax[1].legend(loc="lower left")
ax1_inset = ax[1].inset_axes([0.5, 0.5, 0.4, 0.4])  # [x, y, width, height] 
ax1_inset.plot(t, sigma0)
ax1_inset.set(xlabel="Time [Gyr]", ylabel=r"$\Sigma_0 [{\rm Msun/kpc^2}]$", yscale="log")

ax1_inset1 = ax[1].inset_axes([0.5, 0.1, 0.4, 0.4])  # [x, y, width, height]
ax1_inset1.plot(t, halo_rvir*0.02) 
ax1_inset1.set(xlabel="Time [Gyr]", ylabel=r"$r_{trunc} [{\rm kpc}]$", yscale="log")

# take a ratio of sfr and dot_m_star
ax[2].plot(t, sfr/dot_m_star, label="SFR old / SFR new")
ax[2].plot(t, sfr/classic_dot_m_star, label="SFR old / SFR classic")
ax[2].set(xlabel="Time [Gyr]", ylabel="ratio", yscale="log")
ax[2].legend()

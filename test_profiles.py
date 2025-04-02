# %% testing profiles
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from astropy import cosmology
import scipy
from scipy.interpolate import RegularGridInterpolator

H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


class CoolingFunctionInterpolator:
    def __init__(self, file_path):
        self.file_path = file_path
        self._load_data()
        self.interpolator = RegularGridInterpolator(
            (self.temperatures, self.metallicities),
            self.lambda_values,
            bounds_error=False,
            fill_value=None,
        )

    def _load_data(self):
        # Load the data using numpy, skipping header lines
        data = np.loadtxt(self.file_path, skiprows=5)
        self.temperatures = data[:, 0]
        self.lambda_values = data[:, 1:]

        # Extract metallicities from the header line directly
        with open(self.file_path, "r") as file:
            metallicity_line = file.readlines()[3].strip().split()
            # print(metallicity_line)
            self.metallicities = list(
                map(float, metallicity_line[1 : len(self.lambda_values[0]) + 1])
            )
            # print(self.metallicities)

    def cooling_function(self, temperature, metallicity):
        log_temp = np.log10(temperature)
        log_metallicity = np.log10(metallicity)
        # print(log_temp,  log_metallicity)
        log_lambda = self.interpolator((log_temp, log_metallicity))
        return 10**log_lambda


cooling_fn = CoolingFunctionInterpolator("./tables/newcool_viraj.dat")


def virial_radius(z, mhalo, Delc=200):
    """Halo virial radius, classical 200 topahat overdensity
    Args:
        z (_type_): _description_
        mhalo (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (mhalo / (LCDM.critical_density(z) * (4 / 3) * np.pi * Delc)) ** (1 / 3)


def nfw_prof(r, Mvir, z_red, Delc=200):
    c = 2
    delta_c = (Delc / 3) * (c**3 / (np.log(1 + c) - c / (1 + c)))
    rho_crit_0 = LCDM.critical_density(0)
    print(rho_crit_0)
    r_vir = virial_radius(z_red, Mvir, Delc).to(u.kpc)
    # print(r_vir)
    r_scale = r_vir / c
    # print(r_vir, r_scale)
    numerator = delta_c * rho_crit_0
    rho_dm = numerator / ((r / r_scale) * (1 + (r / r_scale)) ** 2)
    return rho_dm


def rho_gas_0_integrand(x, b):
    return x**2 * (1 + x) ** (27 * b / (2 * x))


def rho_gas_0(gas_frac, rho_crit0, delta_c, b, c):
    prefactor = gas_frac * rho_crit0 * delta_c * np.exp(27 * b / 2)
    prefactor *= np.log(1 + c) - (c / (1 + c))
    integral_result, err = scipy.integrate.quad(rho_gas_0_integrand, 0, c, args=(b))
    return prefactor * integral_result


def rho_gas_makino_cm(r, mcgm_and_ism, Mvir, z_red, halo_temp, Delc=200):
    """gas denisity profile of the CGM mkino et al 2020, dimensionless, g/cm^3"""

    c = 2
    mu = 0.59
    rho_crit_0 = LCDM.critical_density(0)
    gamma = 1.5
    f_gas = mcgm_and_ism / Mvir  # gas fraction of all baryons

    r_vir = virial_radius(z_red, Mvir, Delc).to(u.cm)
    r_scale = r_vir / c
    t_vir = halo_temp * u.K
    delta_c = (Delc / 3) * (c**3 / (np.log(1 + c) - c / (1 + c)))
    b_of_m = (
        8
        * np.pi
        * const.G.value
        * mu
        * const.m_p.value
        * delta_c
        * rho_crit_0.to(u.kg * u.m**-3).value
        * r_scale.to(u.m).value ** 2
    )
    r_scale = r_scale.value  # now turn it into a number after the conversion

    b_of_m /= (27 * const.k_B * t_vir).to(u.m**2 * u.s**-2 * u.kg).value
    # print(b_of_m)

    b_of_r = (
        (2 / (9 * gamma))
        * (r / r_scale)
        * (np.log(1 + r / r_scale) - (r / (r + r_scale))) ** (-1)
    )

    f_baryon = Ob0 / Omegam0
    rho_cgm_0 = rho_gas_0(f_gas, rho_crit_0, delta_c, b_of_m, c).value

    exponent = (27 * b_of_m) / (2 * r / r_scale)
    rho_gas = rho_cgm_0 * np.exp(-27 * b_of_m / 2) * (1 + r / r_scale) ** exponent

    return rho_gas


def cooling_intergrand(
    r, mcgm_and_ism, Mvir, z_red, halo_temp, cooling_lambda, Delc=200
):
    mp = const.m_p.to(u.g)
    mu = 0.6

    profile = rho_gas_makino_cm(
        r, mcgm_and_ism, Mvir, z_red, halo_temp, Delc
    )  # in units of g/cm^3

    number_density = profile / (mu * mp)  # in units of cm^-3
    # print(number_density.unit)
    # print(cooling_lambda.unit)
    integ = number_density.value**2 * cooling_lambda.value * 4 * np.pi * r**2
    # print(integ.unit)
    return integ  # in units of erg / (cm s)


def rho_gas(r, mcgm_and_ism, Mvir, z_red, halo_temp, Delc=200):
    """gas denisity profile of the CGM mkino et al 2020

    Args:
        r (_type_): _description_
        Mvir (_type_): _description_
        z_red (_type_): _description_
        halo_temp (_type_): _description_
        Delc (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: _description_
    """
    c = 2
    mu = 0.59
    rho_crit_0 = LCDM.critical_density(0)
    gamma = 1.5
    f_gas = mcgm_and_ism / Mvir  # gas fraction of all baryons

    r_vir = virial_radius(z_red, Mvir, Delc).to(u.kpc)
    r_scale = r_vir / c
    t_vir = halo_temp * u.K
    delta_c = (Delc / 3) * (c**3 / (np.log(1 + c) - c / (1 + c)))
    b_of_m = (
        8
        * np.pi
        * const.G
        * mu
        * const.m_p
        * delta_c
        * rho_crit_0.to(u.kg * u.m**-3)
        * r_scale.to(u.m) ** 2
    )

    b_of_m /= (27 * const.k_B * t_vir).to(u.m**2 * u.s**-2 * u.kg)
    # print(b_of_m)

    b_of_r = (
        (2 / (9 * gamma))
        * (r / r_scale)
        * (np.log(1 + r / r_scale) - (r / (r + r_scale))) ** (-1)
    )

    f_baryon = Ob0 / Omegam0
    rho_cgm_0 = rho_gas_0(f_gas, rho_crit_0, delta_c, b_of_m, c)

    exponent = (27 * b_of_m) / (2 * r / r_scale)
    rho_gas = rho_cgm_0 * np.exp(-27 * b_of_m / 2) * (1 + r / r_scale) ** exponent

    return rho_gas


T = 1e7
Z = 0.001
cooling_lambda = cooling_fn.cooling_function(T, Z) * (u.erg * u.cm**3 * u.s**-1)

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300)
z = 10
Mvir = 1e12 * u.Msun
mcgm = 1e6 * u.Msun
mism = 1e6 * u.Msun
mcgm_and_ism = mcgm + mism
r_vir = virial_radius(z, Mvir).to(u.kpc)
r = np.geomspace(0.1 * r_vir.value, 1e2 * r_vir.value, 100) * u.kpc
prof = nfw_prof(r, Mvir, z)
gas_prof = rho_gas(r, mism + mcgm, Mvir, z, T)

test = cooling_intergrand(
    r.to(u.cm).value, mcgm_and_ism, Mvir, z, T, cooling_lambda, 200
)

dot_e_cgm_cool, err = scipy.integrate.quad(
    cooling_intergrand,
    0.1 * r_vir.to(u.cm).value,  # Convert r_vir to cm for integration limit
    1e2 * r_vir.to(u.cm).value,  # Convert r_vir to cm for integration limit
    args=(mcgm_and_ism, Mvir, z, T, cooling_lambda, 200),
)
print("the cooling rate is", dot_e_cgm_cool)
ax.plot(r / r_vir, prof)
# ax.plot(r / r_vir, gas_prof)
ax.set(xlabel=r"$r/R_{\rm vir}$", ylabel=r"$\rho(r)$ ", xscale="log", yscale="log")
ax.legend(loc="lower left", fontsize=8, frameon=False)
plt.show()

# %%


def density0(mCGM, r0, Rvir, alpha=1.4):
    denominator = ((4 * np.pi * r0**3) / (3 - alpha)) * ((Rvir / r0) ** (3 - alpha) - 1)
    return mCGM / denominator


def power_profile(r, r1, rho0, alpha=1.4):

    return rho0 * (r / r1) ** (-alpha)


def exponential_disk(r, r_trunc_disk, rho_disk_central):
    """simple exponential disk profile"""
    return rho_disk_central * np.exp(-r / r_trunc_disk)


def disk_density0(mISM, r_trunc_disk, r_vir):
    """
    the total mass of the ism is given analytically by
    (r_trunc_disk - r_trunc_disk* exp(-Rvir/ r_trunc_disk )) rho_0

    here we solve for rho_0

    """
    return mISM / (r_trunc_disk * (1 - np.exp(-r_vir / (r_trunc_disk))))


def nfw_profile(r, rho_bulge, r1):
    return rho_bulge / ((r / r1) * (1 + (r / r1)) ** 2)


def isothermal(r, rho0, r1):
    """isothermal profile"""
    return rho0 / (1 + (r / r1) ** 2)


Rvir = 1
mISM = 1e8  # Msun
mCGM = 1e9  # Msun
r_inner = 0.1  # inner radius of the CGM in Rvir

cgm_alpha = 1.4
rho0 = density0(mCGM, r_inner, Rvir, cgm_alpha)  # central density of the CGM

# radial coordinates
r_full = np.linspace(0, Rvir, 1000)  # units of Rvir
r = np.linspace(0.1, Rvir, 100)  # units of Rvir
r_disk_evaluate = np.linspace(0, 0.1)  # disk scale radius in Rvir
r_trunc_disk = r_inner
rh0_disk = disk_density0(mISM, r_trunc_disk, Rvir)
disk_profile = exponential_disk(r_disk_evaluate, r_trunc_disk, rh0_disk)

# at the radius of the disk, we want the CGM density to match the disk density
matching_disk_density = power_profile(r_inner, r_inner, rho0, cgm_alpha)

# at the truncation radius of the disk
# we want the disk density to match the CGM density
scaling_factor = matching_disk_density / exponential_disk(
    r_inner, r_trunc_disk, rh0_disk
)


# we scale the disk density by this factor and leave the CGM density as is
rh0_disk = rho0 * np.e  # this is now the "true" central density of the disk

# calculate core in terms of the disk truncation
rho_bulge = rho0 * 4 * (Ob0 / Omegam0)

rho_iso = rho0 * 2  # this is the central density of the isothermal profile

# Profiles
disk_profile = exponential_disk(r_disk_evaluate, r_trunc_disk, rh0_disk)
cgm_profile = power_profile(r, r_inner, rho0, 1.4)
bulge_profile = nfw_profile(r_full, rho_bulge, 1)  # bulge profile
isothermal_profile = isothermal(r_full, rho_iso, r_inner)  # isothermal profile
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300)
ax.plot(
    r,
    cgm_profile,
    label=r"power law CGM $\rho_{{\rm CGM}} = \rho_0 (r/r_0)^{{{:.1f}}}$".format(
        cgm_alpha
    ),
    lw=3,
)
ax.plot(
    r_disk_evaluate,
    disk_profile,
    label=r"exponential disk $\rho_{\rm disk} = e \rho_0 \exp(-r/r_0)$",
    lw=3,
)
ax.plot(
    r_full,
    isothermal_profile,
    "--r",
    label=r"isothermal $\rho_{\rm iso} = 2 \rho_0  [1 + (r/r_0)^2]^{-1}$",
    lw=2,
)

ax.plot(
    r_full,
    bulge_profile * (Ob0 / Omegam0),
    label=r"NFW $\times f_b$",
    lw=3,
    color="grey",
    alpha=0.5,
)

ax.set(
    xlabel=r"$r/R_{\rm vir}$",
    ylabel=r"$\rho(r)$ ",
    xscale="log",
    yscale="log",
    xlim=(0.01, 1),
    ylim=(2e7, 8e10),
)
ax.legend(loc="lower left", fontsize=8, frameon=False)


# plt.savefig(
#     "./figures/profile_explorations.png",
#     bbox_inches="tight",
#     dpi=300,
#     transparent=True,
# )
plt.show()

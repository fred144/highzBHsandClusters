import numpy as np
from astropy import cosmology
import h5py
import astropy.units as u
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from cgm_sf_regulator import mhalo_at_z0
from cgm_sf_regulator import CGMRegulator

# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


def run_baseline_model_redshift_grid(observe_at, mhalos, write_to_file=None):
    """
    Runs the baseline CGM regulator model over a grid of redshifts and halo masses.

    For each redshift in `observe_at`, and for each corresponding halo mass in `mhalos`, 
    this function computes the stellar mass, halo mass, and ISM mass using the CGMRegulatorBaseline model.
    The stellar-to-halo mass ratio (SMHM) is calculated and stored for each redshift and halo mass.

    Optionally, results can be saved to an HDF5 file.

    Parameters
    ----------
    observe_at : array-like
        List or array of redshifts at which to observe the halos.
    mhalos : array-like
        2D array (or list of lists) of halo masses. Each sub-array corresponds to the halo masses at a given redshift.
    write_to_file : str or None, optional
        Path to an HDF5 file to write the results. If None, results are not saved to file.

    Returns
    -------
    smhm_out : list of numpy.ndarray
        List of arrays containing the stellar-to-halo mass ratio for each redshift and halo mass.
    zsims : list
        List of redshifts for which the model was run.

    Notes
    -----
    - The function prints progress and intermediate results to stdout.
    - The output file (if specified) contains datasets for redshifts, SMHM, input halo masses, observed halo masses,
      observed stellar masses, and observed ISM masses.
    """
    # declare a 2d array to store the results
    # each row is a different redshift
    # each column is a different halo mass
    smhm_out = []
    zsims = []
    mhalo_obs_out = []
    mstar_obs_out = []
    mism_obs_out = []

    for zidx, z_obs in enumerate(observe_at):
        t_init = 0.1  # Gyr
        # get the final time given the observed redshift
        t_final = LCDM.age(z_obs).value  # Gyr
        print("** t_final = {:.2f}".format(t_final))
        t_span = (t_init, t_final)  # span of the integration

        mhalo_obs = []
        mstar_obs = []
        mism_obs = []

        # now, get the mass of the halo at z0 which is what we put into the integrator
        for midx, mhalo in enumerate(mhalos[zidx]):
            # we want to observe mhalo at z_obs, so we have to know its z=0 value for the function below
            mhalo_z0 = mhalo_at_z0(mhalo, z_obs) * u.Msun
            print("")
            print(
                "* observing halo with mass {:.2e} at z = {:.2f}".format(mhalo, z_obs)
            )
            print("** mass of halo at z=0 is {:.2e}".format(mhalo_z0))
            gridmodel = CGMRegulatorBaseline(
                mhalo_z0,
                t_span,
                cooling_dynamic_time_norm=1,
            )

            run = gridmodel.run_halo()
            results = gridmodel.get_results()
            m_star = results["m_star"][-1]
            m_halo = results["m_halo"][-1]
            print("** final halo mass = {:.2e}".format(m_halo))

            mhalo_obs.append(m_halo)
            mstar_obs.append(m_star)
            mism_obs.append(results["m_gas"][-1])

        mhalo_obs = np.array(mhalo_obs)
        mstar_obs = np.array(mstar_obs)
        mism_obs = np.array(mism_obs)

        # now put it into the output arrays
        mhalo_obs_out.append(mhalo_obs)
        mstar_obs_out.append(mstar_obs)
        mism_obs_out.append(mism_obs)

        smhm = mstar_obs / (mhalo_obs * (Ob0 / Omegam0))

        smhm_out.append(smhm)
        zsims.append(z_obs)

        # save as we go through the redshifts
        print(smhm_out)

    if write_to_file is not None:
        # write to hdf5 file
        smhm_out = np.array(smhm_out)
        zsims = np.array(zsims)
        out_file = h5py.File(write_to_file, "w")
        out_file.create_dataset("redshifts", data=zsims)
        out_file.create_dataset("SMHM", data=smhm_out)
        out_file.create_dataset("Mhalo", data=mhalos)
        out_file.create_dataset("Mhalo_obs", data=mhalo_obs_out)
        # now do star mass, ism mass, cgm hot and cold masses
        out_file.create_dataset("Mstar_obs", data=mstar_obs_out)
        out_file.create_dataset("MISM_obs", data=mism_obs_out)
        out_file.close()

    return smhm_out, zsims


def run_2phase_model_redshift_grid(observe_at, mhalos, write_to_file=None):
    # declare a 2d array to store the results
    # each row is a different redshift
    # each column is a different halo mass
    smhm_out = []
    zsims = []
    mhalo_obs_out = []
    mstar_obs_out = []
    mism_obs_out = []
    mcgm_hot_obs_out = []
    mcgm_cold_obs_out = []
    mbulge_obs_out = []
    sfr_obs_out = []
    m_metals_obs_out = []
    
    # add BH masses
    mbh_obs_out = []
    for zidx, z_obs in enumerate(observe_at):
        t_init = 0.1  # Gyr
        # get the final time given the observed redshift
        t_final = LCDM.age(z_obs).value  # Gyr
        print("** t_final = {:.2f}".format(t_final))
        t_span = (t_init, t_final)  # span of the integration

        mhalo_obs = []
        mstar_obs = []
        mbulge_obs = []
        mism_obs = []
        mcgm_hot_obs = []
        mcgm_cold_obs = []
        mbh_obs = []
        sfr_obs = []
        m_metals_obs = []
        # now, get the mass of the halo at z0 which is what we put into the integrator
        for midx, mhalo in enumerate(mhalos[zidx]):
            # we want to observe mhalo at z_obs, so we have to know its z=0 value for the function below
            mhalo_z0 = mhalo_at_z0(mhalo, z_obs) * u.Msun
            print("")
            print(
                "* observing halo with mass {:.2e} at z = {:.2f}".format(mhalo, z_obs)
            )
            print("** mass of halo at z=0 is {:.2e}".format(mhalo_z0))
            gridmodel = CGMRegulator(
                mhalo_z0, t_span, KS_kappa_s=0.1
            )

            run = gridmodel.run_halo()
            results = gridmodel.get_results()
            derived = gridmodel.get_derived_quantities()
            m_star = results["m_star"][-1]
            m_bulge = results["m_bulge"][-1]
            m_halo = results["m_halo"][-1]
            m_metals = results["m_metals"][-1]
            
            # Z_metal =  m_metals / (results["m_cgm_hot"][-1] + results["m_cgm_cold"][-1])
            # Z_sun = Z_metal / 0.0127

            print("** final halo mass = {:.2e}".format(m_halo))
            mhalo_obs.append(m_halo)
            mstar_obs.append(m_star)
            mbulge_obs.append(m_bulge)
            mism_obs.append(results["m_ism"][-1])
            mcgm_hot_obs.append(results["m_cgm_hot"][-1])
            mcgm_cold_obs.append(results["m_cgm_cold"][-1])
            sfr_obs.append(derived["dot_m_sfr"][-1])
            m_metals_obs.append(m_metals)
            # also get the BH mass

            # mbh_obs.append(results["m_bh"][-1])

        mhalo_obs = np.array(mhalo_obs)
        mstar_obs = np.array(mstar_obs)
        mbulge_obs = np.array(mbulge_obs)
        mism_obs = np.array(mism_obs)
        mcgm_hot_obs = np.array(mcgm_hot_obs)
        mcgm_cold_obs = np.array(mcgm_cold_obs)
        mbh_obs = np.array(mbh_obs)
        # now put it into the output arrays
        mhalo_obs_out.append(mhalo_obs)
        mstar_obs_out.append(mstar_obs)
        mbulge_obs_out.append(mbulge_obs)
        mism_obs_out.append(mism_obs)
        mcgm_hot_obs_out.append(mcgm_hot_obs)
        mcgm_cold_obs_out.append(mcgm_cold_obs)
        sfr_obs_out.append(sfr_obs)
        m_metals_obs_out.append(m_metals_obs)
        # mbh_obs_out.append(mbh_obs)

        smhm = mstar_obs / (mhalo_obs * (Ob0 / Omegam0))

        smhm_out.append(smhm)
        zsims.append(z_obs)

        # save as we go through the redshifts
        print(smhm_out)

    if write_to_file is not None:
        # write to hdf5 file
        smhm_out = np.array(smhm_out)
        zsims = np.array(zsims)
        out_file = h5py.File(write_to_file, "w")
        out_file.create_dataset("redshifts", data=zsims)
        out_file.create_dataset("SMHM", data=smhm_out)
        out_file.create_dataset("Mhalo", data=mhalos)
        out_file.create_dataset("Mhalo_obs", data=mhalo_obs_out)
        # now do star mass, ism mass, cgm hot and cold masses
        out_file.create_dataset("Mstar_obs", data=mstar_obs_out)
        out_file.create_dataset("MISM_obs", data=mism_obs_out)
        out_file.create_dataset("MCGM_hot_obs", data=mcgm_hot_obs_out)
        out_file.create_dataset("MCGM_cold_obs", data=mcgm_cold_obs_out)
        out_file.create_dataset("SFR_obs", data=sfr_obs_out)
        out_file.create_dataset("MMetals_obs", data=m_metals_obs_out)
        # also write the m_bulge
        out_file.create_dataset("MBulge_obs", data=mbulge_obs_out)
        

        # out_file.create_dataset("MBH_obs", data=mbh_obs_out)
        out_file.close()

    return smhm_out, zsims
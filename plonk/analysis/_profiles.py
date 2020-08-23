"""Extra pre-defined profiles."""

from functools import partial

import numpy as np

from .._units import Quantity
from .._units import units as plonk_units

G = (1 * plonk_units.newtonian_constant_of_gravitation).to_base_units()


def extra_profiles(profile, num_separate_dust: int = 0, num_mixture_dust: int = 0):
    """Make extra profiles available.

    Parameters
    ----------
    profile
        The profile object to add extra profiles to.
    num_separate_dust
        The number of "separate sets of particles" dust species.
    num_mixture_dust
        The number of "mixture" dust species.
    """

    @profile.add_profile
    def mass(prof) -> Quantity:
        """Mass profile."""
        M = prof.snap['mass']
        return prof.particles_to_binned_quantity('sum', M)

    @profile.add_profile
    def surface_density(prof) -> Quantity:
        """Surface density profile.

        Units are [mass / length ** ndim], which depends on ndim of profile.
        """
        return prof['mass'] / prof['size']

    @profile.add_profile
    def scale_height(prof) -> Quantity:
        """Scale height profile."""
        z = prof.snap['z']
        return prof.particles_to_binned_quantity('std', z)

    @profile.add_profile
    def aspect_ratio(prof) -> Quantity:
        """Aspect ratio profile."""
        H = prof['scale_height']
        R = prof['radius']
        return H / R

    @profile.add_profile
    def angular_momentum_theta(prof) -> Quantity:
        """Angle between specific angular momentum and xy-plane."""
        angular_momentum_z = prof['angular_momentum_z']
        angular_momentum_magnitude = prof['angular_momentum_mag']

        return np.arccos(angular_momentum_z / angular_momentum_magnitude)

    @profile.add_profile
    def angular_momentum_phi(prof) -> Quantity:
        """Angle between specific angular momentum and x-axis in xy-plane."""
        angular_momentum_x = prof['angular_momentum_x']
        angular_momentum_y = prof['angular_momentum_y']
        return np.arctan2(angular_momentum_y, angular_momentum_x)

    @profile.add_profile
    def toomre_q(prof) -> Quantity:
        """Toomre Q parameter."""
        return (
            prof['sound_speed']
            * prof['keplerian_frequency']
            / (np.pi * G * prof['surface_density'])
        )

    @profile.add_profile
    def alpha_shakura_sunyaev(prof) -> Quantity:
        """Shakura-Sunyaev alpha disc viscosity."""
        try:
            alpha = prof.snap._file_pointer['header/alpha'][()]
        except KeyError:
            raise ValueError('Cannot determine artificial viscosity alpha')
        return alpha / 10 * prof['smoothing_length'] / prof['scale_height']

    @profile.add_profile
    def disc_viscosity(prof) -> Quantity:
        """Disc viscosity. I.e. nu = alpha_SS c_s H."""
        return (
            prof['alpha_shakura_sunyaev'] * prof['sound_speed'] * prof['scale_height']
        )

    @profile.add_profile
    def epicyclic_frequency(prof) -> Quantity:
        """Epicyclic frequency."""
        Omega = prof['keplerian_frequency']
        R = prof['radius']
        return np.sqrt(2 * Omega / R * np.gradient(R ** 2 * Omega, R))

    num_dust_species = num_mixture_dust + num_separate_dust

    for idx in range(num_dust_species):

        def midplane_stokes_number(idx, prof) -> Quantity:
            """Midplane Stokes number profile."""
            gamma = prof.snap.properties['adiabatic_index']
            grain_density = prof.snap.properties['grain_density'][idx]
            grain_size = prof.snap.properties['grain_size'][idx]
            return (
                np.pi
                * np.sqrt(gamma)
                * grain_density
                * grain_size
                / 2
                / prof['surface_density']
            )

        profile._profile_functions[f'midplane_stokes_number_{idx+1:03}'] = partial(
            midplane_stokes_number, idx
        )

    if num_mixture_dust > 0:

        @profile.add_profile
        def gas_mass(prof) -> Quantity:
            """Gas mass profile."""
            M = prof.snap['gas_mass']
            return prof.particles_to_binned_quantity('sum', M)

        @profile.add_profile
        def gas_surface_density(prof) -> Quantity:
            """Gas surface density profile.

            Units are [mass / length ** ndim], which depends on ndim of profile.
            """
            return prof['gas_mass'] / prof['size']

        for idx in range(num_mixture_dust):

            def dust_mass(idx, prof) -> Quantity:
                """Dust mass profile."""
                M = prof.snap[f'dust_mass_{idx+1:03}']
                return prof.particles_to_binned_quantity('sum', M)

            def dust_surface_density(idx, prof) -> Quantity:
                """Dust surface density profile.

                Units are [mass / length ** ndim], which depends on ndim of profile.
                """
                return prof[f'dust_mass_{idx+1:03}'] / prof['size']

            profile._profile_functions[f'dust_mass_{idx+1:03}'] = partial(
                dust_mass, idx
            )

            profile._profile_functions[f'dust_surface_density_{idx+1:03}'] = partial(
                dust_surface_density, idx
            )

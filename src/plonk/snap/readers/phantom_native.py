from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    IO,
    List,
    Literal,
    Tuple,
    Type,
    Union,
    overload,
)

import numpy as np
import pandas as pd

from ..._logging import logger
from ..._units import Quantity
from ..._units import units as plonk_units

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..snap import Snap


def _read_fortran_block(fp: IO, bytesize: int) -> bytes:
    """ Helper function to read Fortran-written data.

    Fortran will add a 4-byte tag before and after any data writes. The value
    of this tag is equal to the number of bytes written. In our case, we do a
    simple sanity check that the start and end tag are consistent, but not
    validate the value of the tag with the size of the data read.
    """
    start_tag = fp.read(4)
    data = fp.read(bytesize)
    end_tag = fp.read(4)

    if (start_tag != end_tag):
        raise AssertionError("Fortran tags mismatch.")

    return data


def _read_capture_pattern(fp: IO) -> Tuple[Type[np.generic],
                                           Type[np.generic],
                                           int]:
    """ Phantom dump validation plus default real and int sizes."""

    start_tag = fp.read(4)  # 4-byte Fortran tag

    def_types: List[Tuple[Type[np.generic],
                          Type[np.generic]]] = [(np.int32, np.float64),
                                                (np.int32, np.float32),
                                                (np.int64, np.float64),
                                                (np.int64, np.float32)]

    i1 = r1 = i2 = 0
    def_int_dtype, def_real_dtype = def_types[0]

    for def_int_dtype, def_real_dtype in def_types:
        i1 = fp.read(def_int_dtype().itemsize)
        r1 = fp.read(def_real_dtype().itemsize)
        i2 = fp.read(def_int_dtype().itemsize)

        i1 = np.frombuffer(i1, count=1, dtype=def_int_dtype)[0]
        r1 = np.frombuffer(r1, count=1, dtype=def_real_dtype)[0]
        i2 = np.frombuffer(i2, count=1, dtype=def_int_dtype)[0]

        if (i1 == def_int_dtype(60769)
                and i2 == def_int_dtype(60878)
                and r1 == def_real_dtype(i2)):
            break
        else:  # rewind and try again
            fp.seek(-def_int_dtype().itemsize, 1)
            fp.seek(-def_real_dtype().itemsize, 1)
            fp.seek(-def_int_dtype().itemsize, 1)

    if (i1 != def_int_dtype(60769)
            or i2 != def_int_dtype(60878)
            or r1 != def_real_dtype(i2)):
        raise AssertionError("Could not determine default int or float "
                             "precision (i1, r1, i2 mismatch). "
                             "Is this a Phantom data file?")

    # iversion -- we don't actually check this
    iversion = fp.read(def_int_dtype().itemsize)
    iversion = np.frombuffer(iversion, count=1, dtype=def_int_dtype)[0]

    # integer 3 == 690706
    i3 = fp.read(def_int_dtype().itemsize)
    i3 = np.frombuffer(i3, count=1, dtype=def_int_dtype)[0]
    if i3 != def_int_dtype(690706):
        raise AssertionError("Capture pattern error. i3 mismatch. "
                             "Is this a Phantom data file?")

    end_tag = fp.read(4)  # 4-byte Fortran tag

    # assert tags equal
    if (start_tag != end_tag):
        raise AssertionError("Capture pattern error. Fortran tags mismatch. "
                             "Is this a Phantom data file?")

    return def_int_dtype, def_real_dtype, iversion


def _read_file_identifier(fp: IO) -> str:
    """ Read the 100 character file identifier.

    The file identifier contains code version and date information.
    """
    return _read_fortran_block(fp, 100).decode('ascii').strip()


def _rename_duplicates(keys: list) -> list:
    seen = dict()

    for i, key in enumerate(keys):
        if key not in seen:
            seen[key] = 1
        else:
            seen[key] += 1
            keys[i] += f'_{seen[key]}'

    return keys


def _read_global_header_block(fp: IO,
                              dtype: Type[np.generic]) -> Tuple[list, list]:
    nvars = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]

    keys = []
    data = []

    if (nvars > 0):
        # each tag is 16 characters in length
        keys_str = _read_fortran_block(fp, 16*nvars).decode('ascii')
        keys = [keys_str[i:i+16].strip() for i in range(0, len(keys_str), 16)]

        raw_data = _read_fortran_block(fp, dtype().itemsize*nvars)
        data = list(np.frombuffer(raw_data, count=nvars, dtype=dtype))

    return keys, data


def _read_global_header(fp: IO,
                        def_int_dtype: Type[np.generic],
                        def_real_dtype: Type[np.generic]) -> dict:
    """ Read global variables. """

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
              def_real_dtype, np.float32, np.float64]

    keys = []
    data = []
    for dtype in dtypes:
        new_keys, new_data = _read_global_header_block(fp, dtype)

        keys += new_keys
        data += new_data

    keys = _rename_duplicates(keys)

    global_vars = dict()
    for i in range(len(keys)):
        global_vars[keys[i]] = data[i]

    return global_vars


def _read_array_block(fp: IO,
                      df: pd.DataFrame,
                      n: int,
                      nums: np.ndarray,
                      def_int_dtype: Type[np.generic],
                      def_real_dtype: Type[np.generic]) -> pd.DataFrame:

    dtypes = [def_int_dtype, np.int8, np.int16, np.int32, np.int64,
              def_real_dtype, np.float32, np.float64]

    for i in range(len(nums)):
        dtype = dtypes[i]
        for j in range(nums[i]):

            tag = _read_fortran_block(fp, 16).decode('ascii').strip()

            if tag in df.columns:
                count = 1
                original_tag = tag
                while tag in df.columns:
                    count += 1
                    tag = original_tag + f"_{count}"

            raw_data = _read_fortran_block(fp, dtype().itemsize * n)
            data: np.ndarray = np.frombuffer(raw_data, dtype=dtype)
            df[tag] = data

    return df


def _read_array_blocks(fp: IO,
                       def_int_dtype: Type[np.generic],
                       def_real_dtype: Type[np.generic]) -> Tuple[
                                                                 pd.DataFrame,
                                                                 pd.DataFrame]:
    """ Read particle data. Block 2 is always for sink particles?"""
    nblocks = np.frombuffer(_read_fortran_block(fp, 4), dtype=np.int32)[0]

    n: List[int] = []
    nums: List[np.ndarray] = []

    for i in range(0, nblocks):
        start_tag = fp.read(4)

        n.append(np.frombuffer(fp.read(8), dtype=np.int64)[0])
        nums.append(np.frombuffer(fp.read(32), count=8, dtype=np.int32))

        end_tag = fp.read(4)
        if (start_tag != end_tag):
            raise AssertionError("Fortran tags mismatch in array blocks.")

    df = pd.DataFrame()
    df_sinks = pd.DataFrame()
    for i in range(0, nblocks):
        # This assumes the second block is only for sink particles.
        # I believe this is a valid assumption as this is what splash assumes.
        # For now we will just append sinks to the end of the data frame.
        if i == 1:
            df_sinks = _read_array_block(fp, df_sinks, n[i], nums[i],
                                         def_int_dtype, def_real_dtype)
        else:
            df = _read_array_block(fp, df, n[i], nums[i], def_int_dtype,
                                   def_real_dtype)

    return df, df_sinks


def _create_mass_column(df: pd.DataFrame,
                        header_vars: dict) -> pd.DataFrame:
    """
    Creates a mass column with the mass of each particle when there are
    multiple itypes.
    """
    df['mass'] = header_vars['massoftype']
    for itype in df['itype'].unique():
        if itype > 1:
            mass = header_vars[f'massoftype_{itype}']
            df.loc[df.itype == itype, 'mass'] = mass
    return df


def _create_aprmass_column(df: pd.DataFrame,
                           header_vars: dict) -> pd.DataFrame:
    """
    Creates a mass column with the mass of each particle when there are
    multiple refinement levels.
    """
    df['mass'] = header_vars['massoftype']
    df['mass'] = df['mass']/(2**(df['apr_level'] - 1))

    return df


def open_native_phantom(filename: str) -> "NativePhantomFile":
    """Read a native Phantom dump into memory for Plonk.

    This uses the parsing helpers in this module and prepares a structure usable
    by Plonk's reader registries.
    """
    with open(filename, 'rb') as fp:
        def_int_dtype, def_real_dtype, iversion = _read_capture_pattern(fp)
        file_identifier = _read_file_identifier(fp)
        header_vars = _read_global_header(fp, def_int_dtype, def_real_dtype)
        header_vars['file_identifier'] = file_identifier
        header_vars['iversion'] = iversion
        header_vars['def_int_dtype'] = def_int_dtype
        header_vars['def_real_dtype'] = def_real_dtype

        df, df_sinks = _read_array_blocks(fp, def_int_dtype, def_real_dtype)

    # Drop inactive particles (negative h) if present
    if 'h' in df.columns:
        df = df[df['h'] > 0]
        # Ensure contiguous positional indexing after filtering
        df = df.reset_index(drop=True)

    # Create a mass column to simplify unit handling later
    if 'itype' in df and df['itype'].nunique() > 1:
        df = _create_mass_column(df, header_vars)
    elif 'apr_level' in df:
        df = _create_aprmass_column(df, header_vars)
    else:
        # Single species - make per-particle mass available as a column
        if 'mass' not in df.columns and 'massoftype' in header_vars:
            df = df.copy()
            df['mass'] = header_vars['massoftype']

    # Make header counts reflect any filtering
    header_vars['nparttot'] = int(df.shape[0])
    header_vars['nptmass'] = int(df_sinks.shape[0]) if isinstance(df_sinks, pd.DataFrame) else 0

    return NativePhantomFile(header=header_vars, particles=df, sinks=df_sinks)


def _compose_vector(df: pd.DataFrame, base: str) -> np.ndarray:
    """Compose a 3-vector array from scalar columns if present.

    base='xyz' -> ['x','y','z']
    base='vxyz' -> ['vx','vy','vz']
    base='Bxyz' -> ['Bx','By','Bz']
    base='spinxyz' -> ['spinx','spiny','spinz'] or ['sx','sy','sz']
    """
    if base == 'xyz':
        cols = ['x', 'y', 'z']
    elif base == 'vxyz':
        cols = ['vx', 'vy', 'vz']
    elif base == 'Bxyz':
        cols = ['Bx', 'By', 'Bz']
    elif base == 'spinxyz':
        cols = ['spinx', 'spiny', 'spinz'] if 'spinx' in df.columns else ['sx', 'sy', 'sz']
    else:
        # Not a recognized vector alias -> signal missing dataset
        raise KeyError(base)
    if all(c in df.columns for c in cols):
        return np.stack([df[c].to_numpy() for c in cols], axis=1)
    raise KeyError(base)


def get_dataset_native(dataset: str, group: str) -> Callable[["Snap"], Quantity]:
    """Return a function that returns an array from NativePhantomFile with units."""

    def func(snap: "Snap") -> Quantity:
        file = snap._file_pointer  # NativePhantomFile
        if group == 'particles':
            df = file.particles
        elif group == 'sinks':
            df = file.sinks
        elif group == 'header':
            # Not used for arrays; treat as scalar values
            val = file.header[dataset]
            return val * plonk_units('dimensionless')
        else:
            raise KeyError(group)

        try:
            if dataset in df.columns:
                array = df[dataset].to_numpy()
            else:
                array = _compose_vector(df, dataset)
        except KeyError:
            if group == 'particles' and dataset == 'itype':
                # Use actual particle count from DataFrame to avoid mismatch with header
                n = df.shape[0]
                logger.debug('Dataset "particles/itype" missing; creating array of ones (gas).')
                array = np.ones(n, dtype=np.int32)
            else:
                raise

        name_map = snap._name_map[group]
        name = name_map.get(dataset, dataset)
        try:
            unit = snap._array_code_units[name]
        except KeyError:
            if group == 'particles' and dataset == 'itype':
                unit = plonk_units('dimensionless')
            else:
                logger.error(
                    f'Cannot get unit of dataset "{group}/{dataset}" - assuming dimensionless'
                )
                unit = plonk_units('dimensionless')
        return array * unit

    return func


def particle_id_native(snap: "Snap") -> Quantity:
    n = snap._file_pointer.particles.shape[0]
    return np.arange(n) * plonk_units('dimensionless')


def particle_type_native(snap: "Snap") -> Quantity:
    idust = snap._file_pointer.header.get('idust', 2)
    ndustlarge = snap._file_pointer.header.get('ndustlarge', 0)
    itype = np.abs(get_dataset_native('itype', 'particles')(snap))
    ptype = itype.copy()
    ptype[(ptype >= idust) & (ptype < idust + ndustlarge)] = snap.particle_type['dust']
    try:
        idustbound = snap._file_pointer.header['idustbound']
        ptype[(ptype >= idustbound) & (ptype < idustbound + ndustlarge)] = snap.particle_type['boundary']
    except KeyError:
        if np.any(ptype >= idust + ndustlarge):
            logger.error('Cannot determine dust boundary particles')
    return np.array(ptype.magnitude, dtype=int) * plonk_units('dimensionless')


def sub_type_native(snap: "Snap") -> Quantity:
    igas, iboundary, istar, idarkmatter, ibulge = 1, 3, 4, 5, 6
    itype = np.abs(get_dataset_native('itype', 'particles')(snap))
    sub_type = np.zeros(itype.shape, dtype=np.int8)
    sub_type[(itype == igas) | (itype == istar) | (itype == idarkmatter) | (itype == ibulge)] = 0
    sub_type[itype == iboundary] = 0
    idust = snap._file_pointer.header.get('idust', 2)
    ndustlarge = snap._file_pointer.header.get('ndustlarge', 0)
    for idx in range(idust, idust + ndustlarge):
        sub_type[itype == idx] = idx - idust + 1
    try:
        idustbound = snap._file_pointer.header['idustbound']
        for idx in range(idustbound, idustbound + ndustlarge):
            sub_type[itype == idx] = idx - idustbound + 1
    except KeyError:
        if np.any(itype >= idust + ndustlarge):
            logger.error('Cannot determine dust boundary particles')
    return sub_type * plonk_units('dimensionless')


def mass_native(snap: "Snap") -> Quantity:
    df = snap._file_pointer.particles
    if 'mass' in df.columns:
        return get_dataset_native('mass', 'particles')(snap)
    # Fallback based on header per-type masses
    itype = np.array(np.abs(get_dataset_native('itype', 'particles')(snap)).magnitude, dtype=int)
    header = snap._file_pointer.header
    mass = np.full(itype.shape, float(header.get('massoftype', 0.0)))
    unique = np.unique(itype)
    for t in unique:
        if t != 1:
            key = f'massoftype_{t}'
            if key in header:
                mass[itype == t] = float(header[key])
    return mass * snap._array_code_units['mass']


def density_native(snap: "Snap") -> Quantity:
    m = (mass_native(snap) / snap._array_code_units['mass']).magnitude
    h = (get_dataset_native('h', 'particles')(snap) / snap._array_code_units['smoothing_length']).magnitude
    hfact = snap.properties['smoothing_length_factor']
    rho = m * (hfact / np.abs(h)) ** 3
    return rho * snap._array_code_units['density']


def pressure_native(snap: "Snap") -> Quantity:
    ieos = snap._file_pointer.header.get('ieos', 2)
    K = 2 / 3 * snap._file_pointer.header.get('RK2', 0.0)
    gamma = snap.properties.get('adiabatic_index', 1.0)
    rho = density_native(snap)
    if ieos == 1:
        K = K * snap.code_units['length'] ** 2 * snap.code_units['time'] ** (-2)
        return K * rho
    if ieos == 2:
        try:
            energy = get_dataset_native('u', 'particles')(snap)
            if gamma > 1.0001:
                return (gamma - 1) * energy * rho
            else:
                return 2 / 3 * energy * rho
        except KeyError:
            K = (
                K * snap.code_units['length'] ** (1 - gamma) * snap.code_units['mass'] ** (-1 + 3 * gamma) * snap.code_units['time'] ** (-2)
            )
            return K * rho ** (gamma - 1)
    if ieos in (3, 6, 14, 21):
        K = K * snap.code_units['length'] ** 2 * snap.code_units['time'] ** (-2)
        q = snap._file_pointer.header.get('qfacdisc', 0.0)
        pos = get_dataset_native('xyz', 'particles')(snap)
        r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2
        r2 = (r2 / snap._array_code_units['position'] ** 2).magnitude
        return K * rho * r2 ** (-q)
    raise ValueError('Unknown equation of state')


def sound_speed_native(snap: "Snap") -> Quantity:
    ieos = snap._file_pointer.header.get('ieos', 2)
    gamma = snap.properties.get('adiabatic_index', 1.0)
    rho = density_native(snap)
    P = pressure_native(snap)
    if ieos in (1, 3, 6, 14, 21):
        return np.sqrt(P / rho)
    if ieos == 2:
        return np.sqrt(gamma * P / rho)
    raise ValueError('Unknown equation of state')


def stopping_time_native(snap: "Snap") -> Quantity:
    bignumber = 1e29
    tstop = get_dataset_native('tstop', 'particles')(snap)
    tstop[tstop == bignumber] = np.inf * snap.code_units['time']
    return tstop


def dust_fraction_native(snap: "Snap") -> Quantity:
    if snap.properties.get('dust_method') != 'dust/gas mixture':
        raise ValueError('Dust fraction only available for "dust/gas mixture"')
    return get_dataset_native('dustfrac', 'particles')(snap)


def dust_to_gas_ratio_native(snap: "Snap") -> Quantity:
    if snap.properties.get('dust_method') != 'dust as separate sets of particles':
        raise ValueError('Dust fraction only available for "dust as separate sets of particles"')
    return get_dataset_native('dustfrac', 'particles')(snap)


def snap_properties_and_units_native(
    file_pointer: "NativePhantomFile",
) -> Tuple[Dict[str, Any], Dict[str, Quantity]]:
    """Generate properties and code units from native header."""
    header = file_pointer.header
    length = (header['udist'] * plonk_units('cm')).to_base_units()
    time = (header['utime'] * plonk_units('s')).to_base_units()
    mass = (header['umass'] * plonk_units('g')).to_base_units()
    temperature = plonk_units('K')
    magnetic_field = (
        header['umagfd'] * plonk_units('g ** (1/2) / cm ** (1/2) / s') * np.sqrt(plonk_units.magnetic_constant / (4 * np.pi))
    ).to_base_units()
    current = (mass / time ** 2 / magnetic_field).to_base_units()

    code_units = {
        'length': length,
        'time': time,
        'mass': mass,
        'temperature': temperature,
        'current': current,
    }

    properties: Dict[str, Any] = {}
    properties['time'] = header['time'] * code_units['time']
    properties['smoothing_length_factor'] = header['hfact']

    gamma = header.get('gamma')
    ieos = header.get('ieos', 2)
    if ieos == 1:
        properties['equation_of_state'] = 'isothermal'
        properties['adiabatic_index'] = gamma
    elif ieos == 2:
        properties['equation_of_state'] = 'adiabatic'
        properties['adiabatic_index'] = gamma
    elif ieos in (3, 6, 14, 21):
        properties['equation_of_state'] = 'locally isothermal disc'
        properties['adiabatic_index'] = gamma

    ndustsmall = header.get('ndustsmall', 0)
    ndustlarge = header.get('ndustlarge', 0)
    if ndustsmall > 0 and ndustlarge > 0:
        raise ValueError(
            'Phantom only supports either dust/gas mixtures (aka 1-fluid dust) or dust as separate sets of particles (aka multi-fluid dust).'
        )
    if ndustsmall > 0:
        properties['dust_method'] = 'dust/gas mixture'
    elif ndustlarge > 0:
        properties['dust_method'] = 'dust as separate sets of particles'

    n_dust = ndustsmall + ndustlarge
    if n_dust > 0:
        properties['grain_size'] = header['grainsize'][:n_dust] * code_units['length']
        properties['grain_density'] = header['graindens'][:n_dust] * code_units['mass'] / code_units['length'] ** 3

    return properties, code_units


def snap_array_registry_native(
    file_pointer: "NativePhantomFile", name_map: Dict[str, str] = None
) -> Dict[str, Callable]:
    """Generate particle array registry for native Phantom data."""
    if name_map is None:
        name_map = {}

    header = file_pointer.header
    arrays = set(file_pointer.particles.columns.tolist())
    ndustsmall = header.get('ndustsmall', 0)
    ndustlarge = header.get('ndustlarge', 0)

    array_registry: Dict[str, Callable] = {}

    # Core arrays
    array_registry['id'] = particle_id_native
    array_registry['type'] = particle_type_native
    array_registry['sub_type'] = sub_type_native
    array_registry['position'] = get_dataset_native('xyz', 'particles')
    array_registry['smoothing_length'] = get_dataset_native('h', 'particles')
    arrays.discard('itype')
    arrays.discard('h')
    for c in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']:
        arrays.discard(c)

    # Dust handling
    if ndustsmall > 0:
        array_registry['dust_fraction'] = dust_fraction_native
        arrays.discard('dustfrac')
    elif ndustlarge > 0:
        array_registry['dust_to_gas_ratio'] = dust_to_gas_ratio_native
        arrays.discard('dustfrac')
    if (ndustsmall > 0 or ndustlarge > 0) and 'tstop' in arrays:
        array_registry['stopping_time'] = stopping_time_native
        arrays.discard('tstop')

    # Mapped arrays if available
    for name_on_file, name in name_map.items():
        if name_on_file in file_pointer.particles.columns or name_on_file in {'xyz', 'vxyz', 'Bxyz'}:
            array_registry[name] = get_dataset_native(name_on_file, 'particles')
            arrays.discard(name_on_file)

    # Derived arrays
    array_registry['mass'] = mass_native
    array_registry['density'] = density_native
    array_registry['pressure'] = pressure_native
    array_registry['sound_speed'] = sound_speed_native

    # Any extra arrays
    for array in sorted(arrays):
        array_registry[array] = get_dataset_native(array, 'particles')

    return array_registry


def snap_sink_registry_native(
    file_pointer: "NativePhantomFile", name_map: Dict[str, str] = None
) -> Dict[str, Callable]:
    """Generate sink array registry for native Phantom data."""
    if name_map is None:
        name_map = {}

    header = file_pointer.header
    if int(header.get('nptmass', 0)) > 0 and isinstance(file_pointer.sinks, pd.DataFrame) and not file_pointer.sinks.empty:
        sinks = set(file_pointer.sinks.columns.tolist())
        sink_registry: Dict[str, Callable] = {}

        for name_on_file, name in name_map.items():
            if name_on_file in file_pointer.sinks.columns or name_on_file in {'xyz', 'vxyz', 'spinxyz'}:
                sink_registry[name] = get_dataset_native(name_on_file, 'sinks')
                sinks.discard(name_on_file)

        for s in sorted(sinks):
            sink_registry[s] = get_dataset_native(s, 'sinks')

        return sink_registry

    return {}


class NativePhantomFile:
    """In-memory wrapper around a native Phantom dump.

    Attributes
    ----------
    header : dict
        Header variables parsed from the dump (keys like 'udist', 'utime', ...).
    particles : pandas.DataFrame
        Particle arrays as columns (e.g. x, y, z, vx, vy, vz, h, itype, ...).
    sinks : pandas.DataFrame
        Sink particle arrays as columns if present.
    """

    def __init__(self, header: Dict[str, Any], particles: pd.DataFrame, sinks: pd.DataFrame):
        self.header = header
        self.particles = particles
        self.sinks = sinks

    def close(self):  # keep Snap.close_file() happy
        return None
"""
Testing Evolution.
"""

import pathlib
import unittest

import numpy as np

import plonk

from .stubdata.phantom_evolution import columns, mean_values

TEST_FILE = pathlib.Path(__file__).parent / 'stubdata/phantom01.ev'


class TestReadPhantomEvolution(unittest.TestCase):
    """Test reading Phantom evolution files."""

    def test_read_evolution(self):
        """Test reading Phantom evolution files."""

        # Read from pathlib.Path
        plonk.Evolution(TEST_FILE)

        # Read from str
        plonk.Evolution(str(TEST_FILE))

        # Not exists
        test_file = 'does_not_exist.ev'
        self.assertRaises(FileNotFoundError, plonk.Evolution, test_file)

    def test_read_evolution_data(self):
        """Test reading data from Phantom evolution files."""

        ev = plonk.Evolution(TEST_FILE)

        self.assertEqual(set(ev.columns), columns)

        for key in ev.columns:
            np.testing.assert_allclose(ev.data[key].mean(), mean_values[key])


class TestPlotPhantomEvolution(unittest.TestCase):
    """Test Phantom evolution files."""

    def test_plot_evolution_data(self):
        """Test plotting data from Phantom evolution files."""

        ev = plonk.Evolution(TEST_FILE)

        ev.plot('xcom')


if __name__ == '__main__':
    unittest.main(verbosity=2)
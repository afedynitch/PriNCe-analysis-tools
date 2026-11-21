"""Fixtures for tests that need a Prince Ultra run object.

They need ``prince_ultra`` and a small FLUKA photo-nuclear db named by the environment variable
``PRINCE_FLUKA_SMOKE_DB`` (e.g. ``tests/data/prince_db_v1_smoke.h5`` of the prince-ultra repository);
without either, the tests that use the fixture are skipped.
"""
import os

import numpy as np
import pytest


@pytest.fixture(scope="session")
def prince_run_m14():
    """Nuclei-only run up to A = 14 on a reduced grid, CPU backend."""
    db = os.environ.get("PRINCE_FLUKA_SMOKE_DB")
    if not db or not os.path.isfile(db):
        pytest.skip("PRINCE_FLUKA_SMOKE_DB does not name a FLUKA db file")
    config = pytest.importorskip("prince_ultra.config")
    config.debug_level = 0
    config.cosmic_ray_grid = (6, 12, 4)
    config.photon_grid = (-12, -8, 4)
    config.x_cut = 1e-4
    config.x_cut_proton = 1e-2
    config.tau_dec_threshold = np.inf
    config.secondaries = False
    config.linear_algebra_backend = "scipy"
    config.fluka_db_path, config.fluka_db_fname = os.path.split(os.path.abspath(db))
    from prince_ultra import core, cross_sections, photonfields

    pf = photonfields.CombinedPhotonField([photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D])
    return core.PriNCeRun(max_mass=14, photon_field=pf, cross_sections=cross_sections.FlukaPhotoNuclear())

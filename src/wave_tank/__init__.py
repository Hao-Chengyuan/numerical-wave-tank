"""wave_tank sub-package — wave generation, kinematics, and visualisation."""

from .wave_tank import IrregularWave_JONSWAP, RegularWave, solve_dispersion

__all__ = ["IrregularWave_JONSWAP", "RegularWave", "solve_dispersion"]

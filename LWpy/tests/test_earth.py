from context import LWpy
import unittest
import EarthModelService
import LeptonInjector

class GeneratorTests(unittest.TestCase):
    """Basic test cases."""

    def test_earth_model_init(self):
        earth_model_params = [
            "DUNE",
            "../resources/earthparams/",
            ["PREM_dune"],
            ["Standard"],
            "NoIce",
            20.0*LeptonInjector.Constants.degrees,
            1480.0*LeptonInjector.Constants.m]
        earthModel = EarthModelService.EarthModelService(*earth_model_params)

    def test_earth_init(self):
        earth_model_params = [
            "DUNE",
            "../resources/earthparams/",
            ["PREM_dune"],
            ["Standard"],
            "NoIce",
            20.0*LeptonInjector.Constants.degrees,
            1480.0*LeptonInjector.Constants.m]
        LWpy.earth(earth_model_params)

    def test_earth_init(self):
        earth_model_params = [
            "DUNE",
            "../resources/earthparams/",
            ["PREM_dune"],
            ["Standard"],
            "NoIce",
            20.0*LeptonInjector.Constants.degrees,
            1480.0*LeptonInjector.Constants.m]
        LWpy.earth(earth_model_params)


if __name__ == '__main__':
    unittest.main()

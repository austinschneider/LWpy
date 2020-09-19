from context import LWpy
import unittest
import LeptonInjector

class GeneratorTests(unittest.TestCase):
    """Basic test cases."""

    def test_generator_init(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        LWpy.generator(blocks[1])

    def test_volume_generator_init(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        LWpy.volume_generator(blocks[1])

    def test_ranged_generator_init(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        earth_model_params = [
            "DUNE",
            "../../LeptonInjectorDUNE/resources/earthparams/",
            ["PREM_dune"],
            ["Standard"],
            "NoIce",
            20.0*LeptonInjector.Constants.degrees,
            1480.0*LeptonInjector.Constants.m]
        LWpy.ranged_generator(blocks[1], earth_model_params)


if __name__ == '__main__':
    unittest.main()

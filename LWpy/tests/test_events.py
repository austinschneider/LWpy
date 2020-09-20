from context import LWpy
import unittest
import LeptonInjector
import numpy as np
import h5py as h5

class EventTests(unittest.TestCase):
    """Basic test cases."""

    def test_ranged_generator_init(self):
        s = LWpy.read_stream('./config_DUNE.lic')
        blocks = s.read()
        earth_model_params = [
            "DUNE",
            "../resources/earthparams/",
            ["PREM_dune"],
            ["Standard"],
            "NoIce",
            20.0*LeptonInjector.Constants.degrees,
            1480.0*LeptonInjector.Constants.m]
        v = blocks[1][2]["height"]
        del blocks[1][2]["height"]
        blocks[1][2]["length"] = v
        gen = LWpy.ranged_generator(blocks[1], earth_model_params)

        data_file = h5.File("data_output_DUNE.h5")
        injector_list = [i for i in data_file.keys()]
        for i in injector_list:
            props = data_file[i]["properties"][:]
            props.dtype.names = (
                    'energy',
                    'zenith',
                    'azimuth',
                    'byorken_x',
                    'byorken_y',
                    'final_state_0',
                    'final_state_1',
                    'particle',
                    'radius',
                    'z',
                    'total_column_depth'
                    )
            names = props.dtype.names
            formats = [(s,v) for s,v in props.dtype.descr if s in names]
            formats += [('x', '<f8'), ('y', '<f8')]
            a = [props[n] for n in names]
            a += [np.zeros(len(props)), np.zeros(len(props))]
            props = np.array(list(zip(*a)), dtype=formats)
            print(gen.prob_pos(props))

if __name__ == '__main__':
    unittest.main()

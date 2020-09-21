from context import LWpy
import unittest
import LeptonInjector
import numpy as np
import h5py as h5

def make_generator():
    s = LWpy.read_stream('./config_DUNE.lic')
    blocks = s.read()
    return s, blocks, LWpy.generator(blocks[1])

def make_volume_generator():
    s = LWpy.read_stream('./config_DUNE.lic')
    blocks = s.read()
    return s, blocks, LWpy.volume_generator(blocks[1])

def make_ranged_generator():
    s = LWpy.read_stream('./config_DUNE.lic')
    blocks = s.read()
    v = blocks[1][2]["height"]
    del blocks[1][2]["height"]
    blocks[1][2]["length"] = v
    earth_model_params = [
        "DUNE",
        "../resources/earthparams/",
        ["PREM_dune"],
        ["Standard"],
        "NoIce",
        20.0*LeptonInjector.Constants.degrees,
        1480.0*LeptonInjector.Constants.m]
    return s, blocks, LWpy.ranged_generator(blocks[1], earth_model_params)

class GeneratorTests(unittest.TestCase):
    """Basic test cases."""

    def test_generator_init(self):
        make_generator()

    def test_generator_stat(self):
        s, blocks, gen = make_generator()
        n0 = gen.prob_stat(None)
        n1 = blocks[1][2]["events"]
        assert(n0 == n1)

    def test_generator_e(self):
        s, blocks, gen = make_generator()
        n = int(1e7)
        e_min = blocks[1][2]["energy_min"]
        e_max = blocks[1][2]["energy_max"]
        events = np.array(np.random.uniform(e_min, e_max, n), dtype=[('energy', 'f8')])
        pe = gen.prob_e(events)
        integral = (np.sum(pe) * (e_max-e_min)/n)
        assert(abs(1.0 - integral) < 0.05)

    def test_generator_dir(self):
        s, blocks, gen = make_generator()
        n = int(1e4)
        block = blocks[1][2]

        zenith_min = block["zenith_min"]
        zenith_max = block["zenith_max"]
        azimuth_min = block["azimuth_min"]
        azimuth_max = block["azimuth_max"]

        zenith = np.random.uniform(zenith_min, zenith_max, n)
        azimuth = np.random.uniform(azimuth_min, azimuth_max, n)

        events = np.array(list(zip(zenith, azimuth)), dtype=[('zenith', 'f8'), ('azimuth', 'f8')])

        probs = gen.prob_dir(events)

        integral = (np.sum(probs) * (azimuth_max-azimuth_min)*(np.cos(zenith_min)-np.cos(zenith_max))/n)
        assert(abs(1.0 - integral) < 0.05)

    def test_generator_final_state(self):
        s, blocks, gen = make_generator()
        n = int(1e4)
        block = blocks[1][2]
        fs_0 = block["final_type_0"]
        fs_1 = block["final_type_1"]

        def get_events():
            return np.array(list(zip(final_type_0, final_type_1)), dtype=[('final_type_0', 'i'), ('final_type_1', 'i')])
        final_type_0 = np.ones(n) * fs_0
        final_type_1 = np.ones(n) * fs_1
        assert(np.sum(gen.prob_final_state(get_events())) == n)

        final_type_0 = np.ones(n) * fs_1
        final_type_1 = np.ones(n) * fs_0
        assert(np.sum(gen.prob_final_state(get_events())) == n)

        final_type_0 = np.ones(n) * (fs_0+1)
        final_type_1 = np.ones(n) * fs_1
        assert(np.sum(gen.prob_final_state(get_events())) == 0)

        final_type_0 = np.ones(n) * fs_0
        final_type_1 = np.ones(n) * fs_1+1
        assert(np.sum(gen.prob_final_state(get_events())) == 0)

    def test_generator_area(self):
        s, blocks, gen = make_generator()
        p = gen.prob_area(None)
        assert(p == 1)

    def test_generator_pos(self):
        s, blocks, gen = make_generator()
        p = gen.prob_pos(None)
        assert(p == 1)

    def test_generator_kinematics(self):
        s, blocks, gen = make_generator()
        n = int(1e6)
        m = int(20)
        block = blocks[1][2]
        e_min = blocks[1][2]["energy_min"]
        e_max = blocks[1][2]["energy_max"]
        energies = np.random.uniform(e_min, e_max, m)
        for energy in energies:
            energy = np.full(n, energy)
            bjorken_x = np.random.uniform(0.0, 1.0, n)
            bjorken_y = np.random.uniform(0.0, 1.0, n)

            events = np.array(list(zip(
                energy,
                bjorken_x,
                bjorken_y)),
                dtype=[('energy', 'f8'), ('bjorken_x', 'f8'), ('bjorken_y', 'f8')])

            p = gen.prob_kinematics(events)
            integral = (np.sum(p)/n)
            assert(abs(1.0 - integral) < 0.05)

    def test_generator_kinematics(self):
        s, blocks, gen = make_generator()
        n = int(1e6)
        block = blocks[1][2]
        e_min = blocks[1][2]["energy_min"]
        e_max = blocks[1][2]["energy_max"]
        fs_0 = block["final_type_0"]
        fs_1 = block["final_type_1"]
        zenith_min = block["zenith_min"]
        zenith_max = block["zenith_max"]
        azimuth_min = block["azimuth_min"]
        azimuth_max = block["azimuth_max"]

        energy = np.random.uniform(e_min, e_max, n)
        final_type_0 = np.ones(n) * fs_0
        final_type_1 = np.ones(n) * fs_1
        zenith = np.random.uniform(zenith_min, zenith_max, n)
        azimuth = np.random.uniform(azimuth_min, azimuth_max, n)
        bjorken_x = np.random.uniform(0.0, 1.0, n)
        bjorken_y = np.random.uniform(0.0, 1.0, n)

        events = np.array(list(zip(
            energy,
            final_type_0,
            final_type_1,
            zenith,
            azimuth,
            bjorken_x,
            bjorken_y)),
            dtype=[
                ('energy', 'f8'),
                ('final_type_0', 'i4'),
                ('final_type_1', 'i4'),
                ('zenith', 'f8'),
                ('azimuth', 'f8'),
                ('bjorken_x', 'f8'),
                ('bjorken_y', 'f8'),
                ])

        gen.prob(events)

    def test_volume_generator_init(self):
        s, blocks, gen = make_volume_generator()

    def test_volume_generator_inside(self):
        s, blocks, gen = make_volume_generator()
        n = int(1e6)
        block = blocks[1][2]
        radius = block["radius"]
        height = block["height"]

        alpha = 2.**(1./3.)
        base_r = alpha * radius
        base_h = alpha * height

        phi = np.random.uniform(0, 2*np.pi, n)
        a = np.random.uniform(0, 1, n)
        r = np.sqrt(a*(base_r**2))

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.random.uniform(-base_h/2., base_h/2., n)
        events = np.array(list(zip(
            x,
            y,
            z)),
            dtype=[
                ('x', 'f8'),
                ('y', 'f8'),
                ('z', 'f8'),
                ])

        integral = float(np.sum(gen.inside_volume(events))) / len(events)
        assert(abs(1.0 - integral*2.) < 0.05)

    def test_volume_generator_area(self):
        s, blocks, gen = make_volume_generator()
        n = int(1e5)
        m = int(20)
        block = blocks[1][2]
        radius = block["radius"]
        height = block["height"]

        alpha = 2.**(1./3.)
        base_r = alpha * radius
        base_h = alpha * height

        phi = np.random.uniform(0, 2*np.pi, n)
        a = np.random.uniform(0, 1, n)
        r = np.sqrt(a*(base_r**2))

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.random.uniform(-base_h/2., base_h/2., n)

        zenith = np.arccos(np.random.uniform(-1, 1, m))
        azimuth = np.random.uniform(0, 2*np.pi, m)
        for az, zen in zip(azimuth, zenith):
            az = np.full(n, az)
            zen = np.full(n, zen)
            events = np.array(list(zip(
                x,
                y,
                z,
                az,
                zen)),
                dtype=[
                    ('x', 'f8'),
                    ('y', 'f8'),
                    ('z', 'f8'),
                    ('azimuth', 'f8'),
                    ('zenith', 'f8'),
                    ])
            mask = gen.inside_volume(events)
            gen.prob_area(events[mask])

    def test_ranged_generator_init(self):
        s, blocks, gen = make_ranged_generator()

    def test_ranged_generator_area(self):
        s, blocks, gen = make_ranged_generator()
        block = blocks[1][2]
        radius = block["radius"]
        prob = gen.prob_area(None)
        integral = (prob * radius*radius*np.pi * 1e4)
        assert(abs(1.0 - integral) < 0.05)

    def test_ranged_generator_pos(self):
        s, blocks, gen = make_ranged_generator()

        data_file = h5.File("data_output_DUNE.h5")
        injector_list = [i for i in data_file.keys()]
        for i in injector_list:
            props = data_file[i]["properties"][:]
            props.dtype.names = (
                    'energy',
                    'zenith',
                    'azimuth',
                    'bjorken_x',
                    'bjorken_y',
                    'final_type_0',
                    'final_type_1',
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
            gen.prob_pos(props)
            gen.prob_area(props)
            gen.get_considered_range(props)

if __name__ == '__main__':
    unittest.main()

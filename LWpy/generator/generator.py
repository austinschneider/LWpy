import numpy as np
import photospline

class generator:
    def __init__(self, block):
        block_name, block_version, block_data = block
        self.block_type = block_name
        self.block_version = block_version
        self.block = block_data
        self.total_xs = photospline.SplineTable(self.block["totalCrossSection"])
        self.differential_xs = photospline.SplineTable(self.block["differentialCrossSection"])
        self.Na = 6.022140857e+23
        self.earth_model = None

    def prob_stat(self, event):
        return self.block["events"]

    def prob_e(self, events):
        norm = 0
        index = self.block["powerlaw_index"]
        energy_min = self.block["energy_min"]
        energy_max = self.block["energy_max"]

        events = np.asarray(events)
        res = np.zeros(len(events))
        energy = events["energy"]
        nonzero = np.logical_and(energy >= energy_min, energy <= energy_max)

        if index != 1:
            norm = (1.0-index) / (energy_max**(1.0-index) - energy_min**(1.0-index))
        elif index == 1:
            norm = 1.0 / np.log(energy_max/energy_min)
        res[nonzero] = norm * energy[nonzero] ** (-index)
        return res

    def prob_dir(self, events):
        events = np.asarray(events)
        res = np.zeros(len(events))
        zenith = events["zenith"]
        azimuth = events["azimuth"]
        nonzero = functools.reduce(np.logical_and,
                [
                zenith >= zenith_min,
                zenith <= zenith_max,
                azimuth >= azimuth_min,
                azimuth <= azimuth_max,
                ])
        res[nonzero] = 1.0/((azimuth_maz-azimuth_min)*(np.cos(zenith_min)-np.cos(zenith_max)))
        return res

    def prob_final_state(self, events):
        events = np.asarray(events)
        final_state_0 = events["final_state_0"]
        final_state_1 = events["final_state_1"]
        return np.logical_or(
            np.logical_and(final_state_0 == self.block["final_state_0"],
                           final_state_1 == self.block["final_state_1"]),
            np.logical_and(final_state_0 == self.block["final_state_1"],
                           final_state_1 == self.block["final_state_0"])
            )

    def prob_area(self, events):
        return 1.0

    def prob_pos(self, events):
        return 1.0

    def prob_interaction(self, events):
        events = np.asarray(events)
        energy = events["energy"]
        x = events["byorken_x"]
        y = events["byorken_y"]
        coords = np.array([energy, x, y])
        return self.differential_xs.evaluate_simple(coords, 0)

    def number_of_targets(self, events):
        events = np.asarray(events)
        return self.Na * events["total_column_depth"]

    def probability(self, events):
        p = self.probability_e(events)
        p *= self.probability_dir(events)
        p *= self.probability_area(events)
        p *= self.probability_pos(events)
        p *= self.probability_final_state(events)
        p *= self.probability_interaction(events)

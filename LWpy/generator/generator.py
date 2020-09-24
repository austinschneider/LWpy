from ..spline import spline_repo, eval_spline
import numpy as np
import photospline

class generator:
    def __init__(self, block):
        block_name, block_version, block_data = block
        self.block_type = block_name
        self.block_version = block_version
        self.block = block_data
        self.total_xs = self.block["totalCrossSection"]
        self.differential_xs = self.block["differentialCrossSection"]
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
        zenith_min = self.block["zenith_min"]
        zenith_max = self.block["zenith_max"]
        azimuth_min = self.block["azimuth_min"]
        azimuth_max = self.block["azimuth_max"]
        zenith = events["zenith"]
        azimuth = events["azimuth"]
        nonzero = np.all(
                [
                zenith >= zenith_min,
                zenith <= zenith_max,
                azimuth >= azimuth_min,
                azimuth <= azimuth_max,
                ], axis=0)
        res[nonzero] = 1.0/((azimuth_max-azimuth_min)*(np.cos(zenith_min)-np.cos(zenith_max)))
        return res

    def prob_final_state(self, events):
        events = np.asarray(events)
        final_type_0 = events["final_type_0"]
        final_type_1 = events["final_type_1"]
        return np.logical_or(
            np.logical_and(final_type_0 == self.block["final_type_0"],
                           final_type_1 == self.block["final_type_1"]),
            np.logical_and(final_type_0 == self.block["final_type_1"],
                           final_type_1 == self.block["final_type_0"])
            ).astype(float)

    def prob_area(self, events):
        raise
        return 1.0

    def prob_pos(self, events):
        raise
        return 1.0

    def prob_kinematics(self, events):
        events = np.asarray(events)
        energy = events["energy"]
        x = events["bjorken_x"]
        y = events["bjorken_y"]
        coords = np.array([np.log10(energy), np.log10(x), np.log10(y)])
        diff_xs = 10.0**eval_spline(spline_repo[self.differential_xs], coords)
        total_xs = 10.0**eval_spline(spline_repo[self.total_xs], coords[:1])
        return diff_xs / total_xs

    def number_of_targets(self, events):
        events = np.asarray(events)
        return self.Na * events["total_column_depth"]

    def prob(self, events):
        p = self.prob_final_state(events)
        p *= self.prob_stat(events)
        nonzero = p != 0
        p[nonzero] *= self.prob_dir(events[nonzero])
        nonzero = p != 0
        p[nonzero] *= self.prob_e(events[nonzero])
        nonzero = p != 0
        p[nonzero] *= self.prob_area(events[nonzero])
        nonzero = p != 0
        p[nonzero] *= self.prob_pos(events[nonzero])
        nonzero = p != 0
        p[nonzero] *= self.prob_kinematics(events[nonzero])
        return p

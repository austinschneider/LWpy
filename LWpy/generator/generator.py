class generator:
    def __init__(self, block):
        self.block_type = block[0]
        self.block = block[1]
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
        nonzero = np.logical_and(events.energy >= energy_min, events.energy <= energy_max)

        if index != 1:
            norm = (1.0-index) / (energy_max**(1.0-index) - energy_min**(1.0-index))
        elif index == 1:
            norm = 1.0 / np.log(energy_max/energy_min)
        res[nonzero] = norm * events[nonzero].energy ** (-index)
        return res

    def prob_dir(self, events):
        events = np.asarray(events)
        res = np.zeros(events.shape)
        nonzero = functools.reduce(np.logical_and,
                [
                events.zenith >= zenith_min,
                events.zenith <= zenith_max,
                events.azimuth >= azimuth_min,
                events.azimuth <= azimuth_max,
                ])
        res[nonzero] = 1.0/((azimuth_maz-azimuth_min)*(np.cos(zenith_min)-np.cos(zenith_max)))
        return res

    def prob_final_state(self, events):
        events = np.asarray(events)
        return np.logical_or(
            np.logical_and(events.final_state_0 == self.block["final_state_0"],
                           events.final_state_1 == self.block["final_state_1"]),
            np.logical_and(events.final_state_0 == self.block["final_state_1"],
                           events.final_state_1 == self.block["final_state_0"])
            )

    def prob_area(self, events):
        return 1.0

    def prob_pos(self, events):
        return 1.0

    def prob_interaction(self, events):
        events = np.asarray(events)
        energy = events.energy
        x = events.byorken_x
        y = events.byorken_y
        coords = np.array([energy, x, y])
        return differential_xs.evaluate_simple(coords, 0)

    def number_of_targets(self, events):
        events = np.asarray(events)
        return self.Na * events.total_column_depth

    def probability(self, events):
        p = self.probability_e(events)
        p *= self.probability_dir(events)
        p *= self.probability_area(events)
        p *= self.probability_pos(events)
        p *= self.probability_final_state(events)
        p *= self.probability_interaction(events)

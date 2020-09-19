class interaction:
    def __init__(self, name, particle, final_state, differential_xs, total_xs):
        self.name = name
        self.differential_xs = differential_xs
        self.total_xs = total_xs
        self.particles = particles
        self.final_state = set(final_state)
        self.signature = (particle, final_state)
        self.key = (name, particle, final_state)

    @classmethod
    def eval_spline(spline, coords, grad=None):
        if np.shape(coords) == 2:
            coords = coords.T
        if grad is None:
            return spline.evaluate_simple(coords, 0)
        else:
            try:
                grad = int(grad)
                assert(grad < len(coords))
                grad = 0x1 << grad
                return spline.evaluate_simple(coords, grad)
            except:
                try:
                    grad = list(grad)
                    res = 0x0
                    for i in range(np.shape(coords)[0]):
                        if i in grad:
                            res |= (0x1 << i)
                    return spline.evaluate_simple(coords, res)
                except:
                    try:
                        return spline.evaluate_simple(coords, grad)
                    except:
                        raise ValueError("Could not interpret grad!")

    def total_cross_section(self, coords, grad=None):
        spline = spline_repo[self.total_xs]
        coords = np.asarray(T)
        return self.eval_spline(spline, coords, grad=grad)

    def differential_cross_section(self, coords, grad=None):
        spline = spline_repo[self.differential_xs]
        coords = np.asarray(T)
        return self.eval_spline(spline, coords, grad=grad)


class interactions:
    def __init__(self, interactions_list=None):
        self.interactions = interactions_list
        if self.interactions is None:
            self.interactions = []
        signatures = [interaction.signature for interaction in interactions]
        assert(len(set(signatures)) == len(signatures))
        self.interactions_by_particle = dict()
        self.interactions_by_final_state = dict()
        self.interactions_by_signature = dict()

        self.interactions_by_name = dict()
        self.interactions_by_key = dict()

        for interaction in self.interactions:
            particle = interaction.particle
            final_state = interaction.final_state
            signature = interaction.signature
            name = interaction.name
            key = interaction.key

            if particle not in self.interactions_by_particle:
                self.interactions_by_particle[particle] = []
            self.interactions_by_particle[particle].append(interaction)

            if final_state not in self.interactions_by_final_state:
                self.interactions_by_final_state[final_state] = []
            self.interactions_by_final_state[final_state].append(interaction)

            if signature not in self.interactions_by_signature:
                self.interactions_by_signature[signature] = []
            self.interactions_by_signature[signature].append()

            if name not in self.interactions_by_name:
                self.interactions_by_name[name] = []
            self.interactions_by_name[name].append()

            self.interactions_by_key[key] = interaction

    def get_particle_interactions(self, particle):
        if particle in self.interactions_by_particle:
            return self.interactions_by_particle[particle]
        else:
            return []

    def get_interactions(self, particle, final_state):
        signature = (particle, set(final_state))
        return self.interactions_by_signature[signature]

    def get_interaction(self, name, particle, final_state):
        key = (name, particle, set(final_state))
        return self.interactions_by_key[key]

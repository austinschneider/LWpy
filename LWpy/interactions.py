from .spline import spline_repo
import numpy as np

class interaction:
    def __init__(self, name, particle, final_state, differential_xs, total_xs):
        self.name = name
        self.differential_xs = differential_xs
        self.total_xs = total_xs
        self.particle = particle
        self.final_state = tuple(sorted(final_state))
        self.signature = (self.particle, self.final_state)
        self.key = (self.name, self.particle, self.final_state)

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
    def __init__(self, interactions_list):
        self.interactions = interactions_list
        signatures = [i.signature for i in self.interactions]
        assert(len(set(signatures)) == len(signatures))
        self.interactions_by_particle = dict()
        self.interactions_by_final_state = dict()
        self.interactions_by_signature = dict()

        self.interactions_by_name = dict()
        self.interactions_by_key = dict()

        for i in self.interactions:
            particle = i.particle
            final_state = i.final_state
            signature = i.signature
            name = i.name
            key = i.key

            if particle not in self.interactions_by_particle:
                self.interactions_by_particle[particle] = []
            self.interactions_by_particle[particle].append(i)

            if final_state not in self.interactions_by_final_state:
                self.interactions_by_final_state[final_state] = []
            self.interactions_by_final_state[final_state].append(i)

            if signature not in self.interactions_by_signature:
                self.interactions_by_signature[signature] = []
            self.interactions_by_signature[signature].append(i)

            if name not in self.interactions_by_name:
                self.interactions_by_name[name] = []
            self.interactions_by_name[name].append(i)

            self.interactions_by_key[key] = i

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

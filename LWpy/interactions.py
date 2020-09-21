from .spline import spline_repo, eval_spline
from .earth import earth
import numpy as np

class interaction:
    def __init__(self, name, particle, final_state, differential_xs, total_xs):
        self.name = name
        self.differential_xs = differential_xs
        self.total_xs = total_xs
        self.particle = int(particle)
        self.final_state = tuple(sorted([int(f) for f in final_state]))
        self.signature = (self.particle, *self.final_state)
        self.key = (self.name, self.particle, *self.final_state)

    def total_cross_section(self, coords, grad=None):
        spline = spline_repo[self.total_xs]
        coords = np.asarray(coords).T
        v = eval_spline(spline, coords, grad=grad)
        return v

    def differential_cross_section(self, coords, grad=None):
        spline = spline_repo[self.differential_xs]
        coords = np.asarray(coords).T
        v = eval_spline(spline, coords, grad=grad)
        return v

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
            particle = int(i.particle)
            final_state = tuple([int(f) for f in i.final_state])
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
        signature = (particle, *final_state)
        return self.interactions_by_signature[signature]

    def get_interaction(self, name, particle, final_state):
        key = (name, particle, *final_state)
        return self.interactions_by_key[key]

class interaction_model(interactions, earth):
    def __init__(self, interactions_list, earth_params):
        interactions.__init__(self, interactions_list)
        earth.__init__(self, earth_params)

    def prob_kinematics(self, events):
        events = np.asarray(events)
        final_state = [events[s] for s in events.dtype.names if "final_state" in s]
        particle = events["particle"]
        energy = events["energy"]
        x = events["bjorken_x"]
        y = events["bjorken_y"]
        coords = np.array([np.log10(energy), np.log10(x), np.log10(y)]).T

        final_state = np.array(final_state).astype(int)
        final_state = np.sort(final_state, axis=0)
        signature = np.concatenate([[particle], final_state]).T
        unique_signatures = np.unique(signature, axis=0)
        unique_signatures_t = [tuple(sig.tolist()) for sig in unique_signatures]
        sig_masks = [np.all(signature == sig[None,:], axis=1) for sig in unique_signatures]
        sig_interactions = [self.get_interactions(sig[0], sig[1:]) for sig in unique_signatures_t]
        diff_xs = np.zeros(len(events)).astype(float)
        total_xs = np.zeros(len(events)).astype(float)
        for sig_mask, relevant_interactions in zip(sig_masks, sig_interactions):
            for i in relevant_interactions:
                diff_xs[sig_mask] += 10.0**(i.differential_cross_section(coords[sig_mask]))
                total_xs[sig_mask] += 10.0**(i.total_cross_section(coords[sig_mask, :1]))
        return diff_xs / total_xs


    def prob_interaction(self, events, first_pos, last_pos):
        segments = self.GetEarthDensitySegments(first_pos, last_pos)

        events = np.asarray(events)
        final_state = [events[s] for s in events.dtype.names if "final_state" in s]
        particle = events["particle"]
        energy = events["energy"]
        x = events["bjorken_x"]
        y = events["bjorken_y"]
        coords = np.array([np.log10(energy), np.log10(x), np.log10(y)]).T

        final_state = np.array(final_state).astype(int)
        final_state = np.sort(final_state, axis=0)
        signature = np.concatenate([[particle], final_state]).T
        unique_signatures = np.unique(signature, axis=0)
        unique_signatures_t = [tuple(sig.tolist()) for sig in unique_signatures]
        sig_masks = [np.all(signature == sig[None,:], axis=1) for sig in unique_signatures]
        sig_interactions = [self.get_interactions(sig[0], sig[1:]) for sig in unique_signatures_t]
        total_xs_by_sig = []

        for sig_mask, relevant_interactions in zip(sig_masks, sig_interactions):
            for i in relevant_interactions:
                total_xs[sig_mask] += 10.0**(i.total_cross_section(coords[sig_mask, :1]))
        return diff_xs / total_xs



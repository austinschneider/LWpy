from .spline import spline_repo
import numpy as np

class interaction:
    def __init__(self, name, particle, final_state, differential_xs, total_xs):
        self.name = name
        self.differential_xs = differential_xs
        self.total_xs = total_xs
        self.particle = particle
        self.final_state = tuple(sorted(final_state))
        self.signature = (self.particle, *self.final_state)
        self.key = (self.name, self.particle, *self.final_state)

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
        signature = (particle, *final_state)
        return self.interactions_by_signature[signature]

    def get_interaction(self, name, particle, final_state):
        key = (name, particle, *final_state)
        return self.interactions_by_key[key]

class interaction_model(interactions, earth):
    def __init__(self, interactions_list, earthparams):
        interactions.__init__(self, interactions_list)
        earth.__init__(self, earth_model_params)

    #GetEarthDensitySegments

    def prob_interaction(self, events):
        events = np.asarray(events)
        final_state = [events[s] for s in events.dtype.names if "final_state" in s]
        particle = events["particle"]

        final_state = np.array(final_state)
        final_state = np.sort(final_state, axis=0)
        signature = np.concatenate([[particle], final_state]).T
        unique_signatures = np.unique(signature, axis=0)
        unique_signatures_t = [tuple(sig.tolist()) for sig in unique_signatures]
        sig_masks = [signature == sig[None,:] for sig in unique_signatures]
        sig_masks = dict(zip(unique_signatures, sig_masks))
        sig_interactions = [self.get_interactions(sig[0], sig[1:]) for sig in unique_signatures_t]
        diff_xs = np.zeros(len(events))
        total_xs = np.zeros(len(events))
        for sig_mask, relevant_interactions in zip(sig_masks, sig_interactions):
            #diff_xs[sig_mask] += 
        interaction_keys = [i.key() for i in si for si in si_interactions]
        interaction_keys = np.unique(interaction_keys, axis=0)
        relevant_interactions = [self.get_interaction(k[0], k[1], k[2:]) for k in interaction_keys]

        for event in events:
            energy = event["energy"]
            x = event["bjorken_x"]
            y = event["bjorken_y"]
            coords = np.array([energy, x, y])


        self.get_interactions()
        energy = events["energy"]
        x = events["bjorken_x"]
        y = events["bjorken_y"]
        coords = np.array([energy, x, y])
        return differential_xs.evaluate_simple(coords, 0)


from .spline import spline_repo, eval_spline
from .earth import earth
import numpy as np
import scipy.special
import LeptonInjector

class interaction:
    def __init__(self, name, particle, final_state, differential_xs, total_xs):
        self.name = name
        self.differential_xs = differential_xs
        self.total_xs = total_xs
        self.particle = int(particle)
        self.final_state = tuple(sorted([int(f) for f in final_state]))
        self.signature = (self.particle, *self.final_state)
        self.key = (self.name, self.particle, *self.final_state)

    def use_electron_density(self):
        return LeptonInjector.getInteraction(
                *[LeptonInjector.Particle.ParticleType(p) for p in self.final_state]) == 2

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
        self.Na = 6.022140857e+23

    def prob_kinematics(self, events):
        events = np.asarray(events)
        final_state = [events[s] for s in events.dtype.names if "final_type" in s]
        particle = events["particle"]
        energy = events["energy"]
        x = events["bjorken_x"]
        y = events["bjorken_y"]
        coords = np.array([np.log10(energy), np.log10(x), np.log10(y)]).T

        final_state = np.array(final_state).astype(int)
        final_state = np.sort(final_state, axis=0)
        signature = np.concatenate([particle[None,:], final_state]).T
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


    def prob_pos(self, events, first_pos, last_pos):
        segments = self.GetDensitySegments(first_pos, last_pos)

        events = np.asarray(events)
        final_state = [events[s] for s in events.dtype.names if "final_state" in s]
        particle = events["particle"]
        energy = events["energy"]
        x = events["bjorken_x"]
        y = events["bjorken_y"]
        coords = np.array([np.log10(energy)]).T
        particles = sorted(np.unique(particle))
        p_indexing = dict(zip(particles, range(len(particles))))
        p_masks = [particle == p for p in particles]
        p_interactions = [self.get_particle_interactions(p) for p in particles]
        p_p_txs = [[10.0**i.total_cross_section(coords[p_mask]) for i in p_int if not i.use_electron_density()] for p_int,p_mask,p in zip(p_interactions,p_masks,particles)]
        p_e_txs = [[10.0**i.total_cross_section(coords[p_mask]) for i in p_int if i.use_electron_density()] for p_int,p_mask,p in zip(p_interactions,p_masks,particles)]
        p_p_txs = [(0 if len(txs)==0 else np.sum(txs, axis=0)) for txs in p_p_txs]
        p_e_txs = [(0 if len(txs)==0 else np.sum(txs, axis=0)) for txs in p_e_txs]

        p_txs_res = np.empty(len(events))
        e_txs_res = np.empty(len(events))
        for p_txs, e_txs, mask in zip(p_p_txs, p_e_txs, p_masks):
            p_txs_res[mask] = p_txs
            e_txs_res[mask] = e_txs

        #total_column_depth_p = self.GetColumnDepthInCGS(first_pos, last_pos, False)
        #total_column_depth_e = self.GetColumnDepthInCGS(first_pos, last_pos, True)

        x = events["x"]
        y = events["y"]
        z = events["z"]
        position = np.array([LeptonInjector.LI_Position(xx, yy, zz) for xx,yy,zz in zip(x,y,z)])

        res = []

        one_m_mexp = lambda val: np.log(val) - val/2. + val**2/24. - val**4/2880. if val < 1e-1 else np.log(1.-np.exp(-val))

        for i, (event, event_segments, p_xs, e_xs, f_p, l_p, pos) in enumerate(zip(events, segments, p_txs_res, e_txs_res, first_pos, last_pos, position)):
            distance = (l_p - f_p) * (pos - f_p)
            total_distance = (l_p - f_p).Magnitude()
            s = []
            exponential_i = []
            exp_i = 0
            segment_distance = 0
            exponentials = []
            got_segment = False
            for segment in event_segments:
                nucleon_density, electron_density, length = segment
                nsigma = self.Na * (p_xs * nucleon_density + e_xs * electron_density) * 1e2 # target interactions per meter
                ss = np.sum(exponentials) + one_m_mexp(nsigma*length) - np.log(nsigma)
                s.append(ss)
                exponential = -nsigma*length
                exponentials.append(exponential)
                segment_distance += length
                if segment_distance > distance and not got_segment:
                    got_segment = True
                    exponential_i = list(exponentials)
                    nsigma_i = nsigma
                    x = distance - (segment_distance - length)
                    exp_i = -nsigma_i*x

            pos_density = np.exp(np.sum(exponential_i) + exp_i - scipy.special.logsumexp(s))
            #nsigmax = self.Na * p_xs * total_column_depth_p[i] + e_xs * total_column_depth_e[i]
            #nsigma = nsigmax / total_distance
            #pos_density_2 = np.exp(-nsigma*distance + np.log(nsigma) - one_m_mexp(nsigmax))
            #pos_density_3 = (1.0 - nsigma*distance)/total_distance
            #print(total_distance, pos_density, pos_density_2/pos_density, pos_density_3/pos_density)
            res.append(pos_density)

        return np.array(res)



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
        v = eval_spline(spline, coords, grad=grad) + 4
        return v

    def differential_cross_section(self, coords, grad=None):
        spline = spline_repo[self.differential_xs]
        coords = np.asarray(coords).T
        v = eval_spline(spline, coords, grad=grad) + 4
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
        if len(events) == 0:
            return np.array(events["particle"].shape)
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

    def get_total_cross_section(self, events):
        events = np.asarray(events)
        if len(events) == 0:
            return np.array(events["particle"].shape), np.array(events["particle"].shape)
        particle = events["particle"]
        energy = events["energy"]
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

        return p_txs_res, e_txs_res

    def get_final_state_cross_section(self, events):
        events = np.asarray(events)
        if len(events) == 0:
            return np.array(events["particle"].shape), np.array(events["particle"].shape)
        particle = events["particle"]
        energy = events["energy"]
        coords = np.array([np.log10(energy)]).T
        particles = sorted(np.unique(particle))
        final_state = [events[s] for s in events.dtype.names if "final_type" in s]
        final_state = np.array(final_state).astype(int)
        final_state = np.sort(final_state, axis=0)
        signature = np.concatenate([particle[None,:], final_state]).T
        unique_signatures = np.unique(signature, axis=0)
        unique_signatures_t = [tuple(sig.tolist()) for sig in unique_signatures]
        sig_masks = [np.all(signature == sig[None,:], axis=1) for sig in unique_signatures]
        sig_interactions = [self.get_interactions(sig[0], sig[1:]) for sig in unique_signatures_t]
        s_p_txs = [[10.0**i.total_cross_section(coords[s_mask]) for i in s_int if not i.use_electron_density()] for s_int,s_mask,s in zip(sig_interactions,sig_masks,signature)]
        s_e_txs = [[10.0**i.total_cross_section(coords[s_mask]) for i in s_int if i.use_electron_density()] for s_int,s_mask,s in zip(sig_interactions,sig_masks,signature)]

        s_p_txs = [(0 if len(txs)==0 else np.sum(txs, axis=0)) for txs in s_p_txs]
        s_e_txs = [(0 if len(txs)==0 else np.sum(txs, axis=0)) for txs in s_e_txs]

        p_txs_res = np.empty(len(events))
        e_txs_res = np.empty(len(events))
        for p_txs, e_txs, mask in zip(s_p_txs, s_e_txs, sig_masks):
            p_txs_res[mask] = p_txs
            e_txs_res[mask] = e_txs

        return p_txs_res, e_txs_res


    @staticmethod
    def log_one_m_mexp(val):
        return np.log(val) - val/2. + val**2/24. - val**4/2880. if val < 1e-1 else np.log(1.-np.exp(-val))

    @staticmethod
    def one_m_mexp(val):
        val = np.asarray(val)
        mask = val < 1e-1
        res = np.empty(np.shape(val))
        val_less, val_greater = val[mask], val[~mask]
        res[mask] = np.exp(np.log(val_less) - val_less/2. + val_less**2/24. - val_less**4/2880.)
        res[~mask] = 1.-np.exp(-val_greater)
        return res

    def prob_pos(self, events, first_pos, last_pos):
        if len(events) == 0:
            return np.array(events["particle"].shape)
        segments = list(reversed(self.GetDensitySegments(last_pos, first_pos)))

        p_txs_res, e_txs_res = self.get_total_cross_section(events)

        #total_column_depth_p = self.GetColumnDepthInCGS(last_pos, first_pos, False)
        #total_column_depth_e = self.GetColumnDepthInCGS(last_pos, first_pos, True)

        x = events["x"]
        y = events["y"]
        z = events["z"]
        position = np.array([LeptonInjector.LI_Position(xx, yy, zz) for xx,yy,zz in zip(x,y,z)])

        res = []

        for i, (event, event_segments, p_xs, e_xs, f_p, l_p, pos) in enumerate(zip(events, segments, p_txs_res, e_txs_res, first_pos, last_pos, position)):
            distance = (pos - f_p).Magnitude()
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
                if nsigma == 0:
                    pass
                    ss = np.sum(exponentials) - np.log(length)
                else:
                    ss = np.sum(exponentials) + self.log_one_m_mexp(nsigma*length) - np.log(nsigma)
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
            #pos_density_2 = np.exp(-nsigma*distance + np.log(nsigma) - self.log_one_m_mexp(nsigmax))
            #pos_density_3 = (1.0 - nsigma*distance)/total_distance
            #print(total_distance, pos_density, pos_density_2/pos_density, pos_density_3/pos_density)
            res.append(pos_density)

        return np.array(res)

    def prob_interaction(self, events, first_pos, last_pos):
        events = np.asarray(events)
        if len(events) == 0:
            return np.array(events["particle"].shape)

        total_column_depth_p = self.GetColumnDepthInCGS(first_pos, last_pos, False)
        total_column_depth_e = self.GetColumnDepthInCGS(first_pos, last_pos, True)

        p_txs, e_txs = self.get_total_cross_section(events)

        exponent = self.Na * (p_txs * total_column_depth_p + e_txs * total_column_depth_e)
        #print(total_column_depth_p / events["total_column_depth"])
        return self.one_m_mexp(exponent)

    def prob_final_state(self, events):
        if len(events) == 0:
            return np.array(events["particle"].shape)
        x = events["x"]
        y = events["y"]
        z = events["z"]
        position = np.array([LeptonInjector.LI_Position(xx, yy, zz) for xx,yy,zz in zip(x,y,z)])

        p_density = self.GetDensityInCGS(position)
        e_density = p_density * self.GetPNERatio(position)

        p_txs, e_txs = self.get_total_cross_section(events)
        p_fs_txs, e_fs_txs = self.get_final_state_cross_section(events)
        return (p_fs_txs * p_density + e_fs_txs * e_density) / (p_txs * p_density + e_txs * e_density)

import pathlib
path = str(pathlib.Path(__file__).parent.absolute())
path += '/resources/'

def get_standard_interactions():
    name = "CC"
    particle = LeptonInjector.Particle.ParticleType.NuE
    final_state = []
    final_state.append(LeptonInjector.Particle.ParticleType.EPlus)
    final_state.append(LeptonInjector.Particle.ParticleType.Hadrons)

    cc_nu_differential_xs = path + "crosssections/csms_differential_v1.0/dsdxdy_nu_CC_iso.fits"
    cc_nu_total_xs = path + "crosssections/csms_differential_v1.0/sigma_nu_CC_iso.fits"
    nc_nu_differential_xs = path + "crosssections/csms_differential_v1.0/dsdxdy_nu_NC_iso.fits"
    nc_nu_total_xs = path + "crosssections/csms_differential_v1.0/sigma_nu_NC_iso.fits"
    cc_nubar_differential_xs = path + "crosssections/csms_differential_v1.0/dsdxdy_nubar_CC_iso.fits"
    cc_nubar_total_xs = path + "crosssections/csms_differential_v1.0/sigma_nubar_CC_iso.fits"
    nc_nubar_differential_xs = path + "crosssections/csms_differential_v1.0/dsdxdy_nubar_NC_iso.fits"
    nc_nubar_total_xs = path + "crosssections/csms_differential_v1.0/sigma_nubar_NC_iso.fits"

    neutrinos = [
            LeptonInjector.Particle.ParticleType.NuE,
            LeptonInjector.Particle.ParticleType.NuMu,
            LeptonInjector.Particle.ParticleType.NuTau,
            LeptonInjector.Particle.ParticleType.NuEBar,
            LeptonInjector.Particle.ParticleType.NuMuBar,
            LeptonInjector.Particle.ParticleType.NuTauBar,
            ]
    charged_leptons = [
            LeptonInjector.Particle.ParticleType.EMinus,
            LeptonInjector.Particle.ParticleType.MuMinus,
            LeptonInjector.Particle.ParticleType.TauMinus,
            LeptonInjector.Particle.ParticleType.EPlus,
            LeptonInjector.Particle.ParticleType.MuPlus,
            LeptonInjector.Particle.ParticleType.TauPlus,
            ]
    cc_total_xs = [cc_nu_total_xs]*3 +[cc_nubar_total_xs]*3
    nc_total_xs = [nc_nu_total_xs]*3 +[nc_nubar_total_xs]*3
    cc_differential_xs = [cc_nu_differential_xs]*3 +[cc_nubar_differential_xs]*3
    nc_differential_xs = [nc_nu_differential_xs]*3 +[nc_nubar_differential_xs]*3

    interactions_list = []
    for nu, lep, cc_txs, cc_xs, nc_txs, nc_xs in zip(neutrinos, charged_leptons, cc_total_xs, cc_differential_xs, nc_total_xs, nc_differential_xs):
        cc_final_state = [lep, LeptonInjector.Particle.ParticleType.Hadrons]
        cc = interaction("CC", nu, cc_final_state, cc_xs, cc_txs)

        nc_final_state = [nu, LeptonInjector.Particle.ParticleType.Hadrons]
        nc = interaction("NC", nu, nc_final_state, nc_xs, nc_txs)

        interactions_list.extend([cc, nc])

    return interactions_list

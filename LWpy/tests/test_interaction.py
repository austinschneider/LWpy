from context import LWpy
from context import standard_interactions
import unittest
import LeptonInjector

class InteractionTests(unittest.TestCase):
    """Basic test cases."""

    def test_interaction_init(self):
        name = "CC"
        particle = LeptonInjector.Particle.ParticleType.NuE
        final_state = []
        final_state.append(LeptonInjector.Particle.ParticleType.EMinus)
        final_state.append(LeptonInjector.Particle.ParticleType.Hadrons)
        differential_xs = "../resources/crosssections/csms_differential_v1.0/dsdxdy_nu_CC_iso.fits"
        total_xs = "../resources/crosssections/csms_differential_v1.0/sigma_nu_CC_iso.fits"
        LWpy.interaction(name, particle, final_state, differential_xs, total_xs)

    def test_interactions_init(self):
        name = "CC"
        particle = LeptonInjector.Particle.ParticleType.NuE
        final_state = []
        final_state.append(LeptonInjector.Particle.ParticleType.EMinus)
        final_state.append(LeptonInjector.Particle.ParticleType.Hadrons)
        differential_xs = "../resources/crosssections/csms_differential_v1.0/dsdxdy_nu_CC_iso.fits"
        total_xs = "../resources/crosssections/csms_differential_v1.0/sigma_nu_CC_iso.fits"
        i = LWpy.interaction(name, particle, final_state, differential_xs, total_xs)
        LWpy.interactions([i])

    def test_standard_interactions(self):
        nu_interactions_list = standard_interactions.get_standard_interactions()
        nu_interactions = LWpy.interactions(nu_interactions_list)
        ints = nu_interactions.get_particle_interactions(LeptonInjector.Particle.ParticleType.NuEBar)
        print([i.name for i in ints])

#    def get_particle_interactions(self, particle):
#        if particle in self.interactions_by_particle:
#            return self.interactions_by_particle[particle]
#        else:
#            return []
#
#    def get_interactions(self, particle, final_state):
#        signature = (particle, set(final_state))
#        return self.interactions_by_signature[signature]
#
#    def get_interaction(self, name, particle, final_state):
#        key = (name, particle, set(final_state))
#        return self.interactions_by_key[key]

if __name__ == '__main__':
    unittest.main()

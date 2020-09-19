from context import LWpy
import LeptonInjector
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())

def get_standard_interactions():
    name = "CC"
    particle = LeptonInjector.Particle.ParticleType.NuE
    final_state = []
    final_state.append(LeptonInjector.Particle.ParticleType.EPlus)
    final_state.append(LeptonInjector.Particle.ParticleType.Hadrons)
    cc_nu_differential_xs = path + "../crosssections/csms_differential_v1.0/dsdxdy_nu_CC_iso.fits"
    cc_nu_total_xs = path + "../crosssections/csms_differential_v1.0/sigma_nu_CC_iso.fits"
    nc_nu_differential_xs = path + "../crosssections/csms_differential_v1.0/dsdxdy_nu_NC_iso.fits"
    nc_nu_total_xs = path + "../crosssections/csms_differential_v1.0/sigma_nu_NC_iso.fits"
    cc_nubar_differential_xs = path + "../crosssections/csms_differential_v1.0/dsdxdy_nubar_CC_iso.fits"
    cc_nubar_total_xs = path + "../crosssections/csms_differential_v1.0/sigma_nubar_CC_iso.fits"
    nc_nubar_differential_xs = path + "../crosssections/csms_differential_v1.0/dsdxdy_nubar_NC_iso.fits"
    nc_nubar_total_xs = path + "../crosssections/csms_differential_v1.0/sigma_nubar_NC_iso.fits"

    neutrinos = [
            LeptonInjector.Particle.ParticleType.NuE,
            LeptonInjector.Particle.ParticleType.NuMu,
            LeptonInjector.Particle.ParticleType.NuTau,
            LeptonInjector.Particle.ParticleType.NuEBar,
            LeptonInjector.Particle.ParticleType.NuMuBar,
            LeptonInjector.Particle.ParticleType.NuTauBar,
            ]
    charged_leptons = [
            LeptonInjector.Particle.ParticleType.EPlus,
            LeptonInjector.Particle.ParticleType.MuPlus,
            LeptonInjector.Particle.ParticleType.TauPlus,
            LeptonInjector.Particle.ParticleType.EMinus,
            LeptonInjector.Particle.ParticleType.MuMinus,
            LeptonInjector.Particle.ParticleType.TauMinus,
            ]
    cc_total_xs = [cc_nu_total_xs]*3 +[cc_nubar_total_xs]*3
    nc_total_xs = [nc_nu_total_xs]*3 +[nc_nubar_total_xs]*3
    cc_differential_xs = [cc_nu_differential_xs]*3 +[cc_nubar_differential_xs]*3
    nc_differential_xs = [nc_nu_differential_xs]*3 +[nc_nubar_differential_xs]*3

    interactions_list = []
    for nu, lep, cc_txs, cc_xs, nc_txs, nc_xs in zip(neutrinos, charged_leptons, cc_total_xs, cc_differential_xs, nc_total_xs, nc_differential_xs):
        cc_final_state = [lep, LeptonInjector.Particle.ParticleType.Hadrons]
        cc = LWpy.interaction("CC", nu, cc_final_state, cc_xs, cc_txs)

        nc_final_state = [nu, LeptonInjector.Particle.ParticleType.Hadrons]
        nc = LWpy.interaction("NC", nu, nc_final_state, nc_xs, nc_txs)

        interactions_list.extend([cc, nc])

    standard_interactions = LWpy.interactions(interactions_list)
    return standard_interactions


from .generator import generator
from ..earth import earth
import numpy as np
import functools
import EarthModelService
import LeptonInjector

class ranged_generator(generator, earth):
    def __init__(self, block, earth_model_params=None, spline_dir='./'):
        generator.__init__(self, block, spline_dir=spline_dir)
        earth.__init__(self, earth_model_params)

    def prob_area(self, events):
        events = np.asarray(events)
        radius = self.block["radius"]
        p_area = 1.0 / (np.pi * radius * radius)
        p_area /= 1e4 # Convert from m^-2 to cm^-2
        return p_area

    def get_considered_range(self, events):
        energy = events["energy"]
        zenith = events["zenith"]
        azimuth = events["azimuth"]
        x = events["x"]
        y = events["y"]
        z = events["z"]
        isTau = ((self.block["final_type_0"] == LeptonInjector.Particle.ParticleType.TauMinus
            or self.block["final_type_0"] == LeptonInjector.Particle.ParticleType.TauPlus)
            or (self.block["final_type_1"] == LeptonInjector.Particle.ParticleType.TauMinus
                or self.block["final_type_1"] == LeptonInjector.Particle.ParticleType.TauPlus))

        use_electron_density = LeptonInjector.getInteraction(
                    LeptonInjector.Particle.ParticleType(self.block["final_type_0"]),
                    LeptonInjector.Particle.ParticleType(self.block["final_type_1"])) == 2

        position = np.array([LeptonInjector.LI_Position(xx, yy, zz) for xx,yy,zz in zip(x,y,z)])
        direction = np.array([LeptonInjector.LI_Direction(zen, azi) for zen,azi in zip(zenith, azimuth)])
        endcapLength = self.block["length"] * LeptonInjector.Constants.meter

        pca = self.get_pca(direction, position)

        lepton_range = self.MWEtoColumnDepthCGS(self.GetLeptonRange(energy, isTau=isTau))
        endcap_range = self.GetColumnDepthInCGS(
                pca - endcapLength*direction,
                pca + endcapLength*direction,
                use_electron_density)

        totalColumnDepth = lepton_range + endcap_range

        maxDist = self.DistanceForColumnDepthToPoint(
                pca + endcapLength*direction,
                direction,
                totalColumnDepth,
                use_electron_density) - endcapLength

        actualColumnDepth = self.GetColumnDepthInCGS(
                pca + endcapLength * direction,
                pca - maxDist * direction,
                use_electron_density)

        mask = actualColumnDepth < (totalColumnDepth-1)

        totalColumnDepth[mask] = actualColumnDepth[mask]

        first_point = pca - maxDist * direction
        last_point = pca + endcapLength * direction

        return first_point, last_point

    def prob_pos(self, events):
        events = np.asarray(events)

        x = events["x"]
        y = events["y"]
        z = events["z"]

        use_electron_density = LeptonInjector.getInteraction(
                    LeptonInjector.Particle.ParticleType(self.block["final_type_0"]),
                    LeptonInjector.Particle.ParticleType(self.block["final_type_1"])) == 2

        first_pos, last_pos = self.get_considered_range(events)

        position = np.array([LeptonInjector.LI_Position(xx, yy, zz) for xx,yy,zz in zip(x,y,z)])

        totalColumnDepth = self.GetColumnDepthInCGS(last_pos, first_pos, use_electron_density)

        density = self.GetDensityInCGS(position)
        assert(np.all(density > 0))
        if use_electron_density:
            density *= self.GetPNERatio(position)

        res = density / totalColumnDepth * 1e2 # Convert cm^-1 to m^-1
        return res


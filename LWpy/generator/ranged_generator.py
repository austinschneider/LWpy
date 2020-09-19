from .generator import generator
import numpy as np
import functools
import EarthModelService
import LeptonInjector

class ranged_generator(generator):
    def __init__(self, block, earth_model_params=None):
        generator.__init__(self, block)
        self.earthModel = EarthModelService.EarthModelService(*earth_model_params)

    @staticmethod
    @np.vectorize
    def get_pca(direction, position, origin=(0,0,0)):
        origin = LeptonInjector.LI_Position(*origin)
        return (direction*(origin - position)) * direction + position

    @staticmethod
    @np.vectorize
    def GetLeptonRange(energy,
                      isTau=False,
                      option=EarthModelService.EarthModelCalculator.LeptonRangeOption.DEFAULT,
                      scale=1.):
        return EarthModelService.EarthModelCalculator.GetLeptonRange(energy, isTau, option, scale)

    @staticmethod
    @np.vectorize
    def MWEtoColumnDepthCGS(range_MWE):
        return EarthModelService.EarthModelCalculator.MWEtoColumnDepthCGS(range_MWE)

    @staticmethod
    @np.vectorize
    def ColumnDepthCGStoMWE(cdep_CGS):
        return EarthModelService.EarthModelCalculator.ColumnDepthCGStoMWE(cdep_CGS)

    @staticmethod
    @np.vectorize
    def GetAtmoPoints(pca, direction):
        atmoEntry = LeptonInjector.LI_Position()
        atmoExit = LeptonInjector.LI_Position()
        EarthModelService.EarthModelCalculator.GetIntersectionsWithSphere(
                earthModel.GetEarthCoordPosFromDetCoordPos(pca),
                earthModel.GetEarthCoordDirFromDetCoordDir(direction),
                earthModel.GetAtmoRadius(), atmoEntry, atmoExit)

        atmoEntry = earthModel.GetDetCoordPosFromEarthCoordPos(atmoEntry)
        atmoExit = earthModel.GetDetCoordPosFromEarthCoordPos(atmoExit)
        return atmoEntry, atmoExit

    def prob_area(self, events):
        events = np.asarray(events)
        length = self.chord_length(events)
        radius = self.block["radius"]
        p_area = 1.0 / (np.pi * radius * radius)
        p_area /= 1e4 # Convert from m^-2 to cm^-2
        return p_area

    def prob_pos(self, events):
        def print_pos(pos):
            print(pos.GetX(), pos.GetY(), pos.GetZ())

        energy = events.energy
        isTau = ((self.block.final_type_1 == LeptonInjector.Particle.ParticleType.TauMinus
                 or self.block.final_type_1 == LeptonInjector.Particle.ParticleType.TauPlus)
                 or (self.block.final_type_2 == LeptonInjector.Particle.ParticleType.TauMinus
                 or self.block.final_type_2 == LeptonInjector.Particle.ParticleType.TauPlus))

        use_electron_density = LeptonInjector.getInteraction(
            self.block["final_type_1"],
            self.block["final_type_2"])

        position = np.array([LeptonInjector.LI_Position(x, y, z) for x,y,z in zip(events.x, events.y, events.z)])
        direction = np.array([LeptonInjector.LI_Direction(zenith, azimuth) in zenith,azimuth in zip(events.zenith, events.azimuth)])
        endcapLength = self.block["length"] * LeptonInjector.Constants.meter

        pca = self.get_pca(direction, position)

        lepton_range = MWEtoColumnDepthCGS(GetLeptonRange(events.energy, isTau=isTau))
        endcap_range = earthModel.GetColumnDepthInCGS(
                pca - endcapLength*direction,
                pca + endcapLength*direction,
                use_electron_density)

        totalColumnDepth = lepton_range + endcap_range

        maxDist = earthModel.DistanceForColumnDepthToPoint(
            pca + endcapLength*direction,
            direction,
            totalColumnDepth,
            use_electron_density) - endcapLength

        actualColumnDepth = earthModel.GetColumnDepthInCGS(
            pca + endcapLength * direction,
            pca - maxDist * direction,
            use_electron_density)


        totalColumnDepth[actualColumnDepth < (totalColumnDepth-1)] = actualColumnDepth

        first_point = pca - maxDist * direction
        last_point = pca + endcapLength * direction

        atmoEntry, atmoExit = GetAtmoPoints(pca, direction, atmoEntry, atmoExit)

        atmoDist = np.array([x.Magnitude() for x in (pca-atmoEntry)])
        dist = np.array([x.Magnitude() for x in (pca - first_point)])
        if np.abs(dist-atmoDist)<100.0:
            dist = min(dist,atmoDist)

        segments = earthModel.GetEarthDensitySegments(first_point, last_point, use_electron_density)
        #rint((first_point - last_point).Magnitude())

        #earthModel.GetColumnDepthInCGS
        #earthModel.GetColumnDepthInCGS(pca-config.endcapLength*dir,pca+config.endcapLength*dir, use_electron_density)

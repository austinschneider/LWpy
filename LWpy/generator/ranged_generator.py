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
    def GetLeptonRange(energy,
            isTau=False,
            option=EarthModelService.EarthModelCalculator.LeptonRangeOption.DEFAULT,
            scale=1.):
        return np.array([EarthModelService.EarthModelCalculator.GetLeptonRange(e, isTau, option, scale) for e in energy])

    @staticmethod
    @np.vectorize
    def MWEtoColumnDepthCGS(range_MWE):
        return EarthModelService.EarthModelCalculator.MWEtoColumnDepthCGS(range_MWE)

    @staticmethod
    @np.vectorize
    def ColumnDepthCGStoMWE(cdep_CGS):
        return EarthModelService.EarthModelCalculator.ColumnDepthCGStoMWE(cdep_CGS)

    def GetAtmoPoints(self, pca, direction):
        a_entry = []
        a_exit = []
        for pca, direction in zip(pca, direction):
            atmoEntry = None
            atmoExit = None
            atmoEntry = LeptonInjector.LI_Position()
            atmoExit = LeptonInjector.LI_Position()
            EarthModelService.EarthModelCalculator.GetIntersectionsWithSphere(
                    self.earthModel.GetEarthCoordPosFromDetCoordPos(pca),
                    self.earthModel.GetEarthCoordDirFromDetCoordDir(direction),
                    self.earthModel.GetAtmoRadius(), atmoEntry, atmoExit)

            atmoEntry = self.earthModel.GetDetCoordPosFromEarthCoordPos(atmoEntry)
            atmoExit = self.earthModel.GetDetCoordPosFromEarthCoordPos(atmoExit)
            a_entry.append(atmoEntry)
            a_exit.append(atmoExit)
        return np.array(a_entry), np.array(a_exit)

    def GetColumnDepthInCGS(self, p0, p1, use_electron_density=False):
        return np.array([self.earthModel.GetColumnDepthInCGS(
            pp0,
            pp1,
            use_electron_density)
            for pp0,pp1 in zip(p0, p1)])


    def DistanceForColumnDepthToPoint(self, p0, d0, col, use_electron_density=False):
        return np.array([self.earthModel.DistanceForColumnDepthToPoint(
            p,
            d,
            c,
            use_electron_density)
            for p,d,c in zip(p0, d0, col)])
    def GetEarthDensitySegments(self, first_point, last_point, use_electron_density=False):
        return [self.earthModel.GetEarthDensitySegments(fp, lp, use_electron_density) for fp,lp in zip(first_point, last_point)]

    def GetEarthDensityInCGS(self, position):
        return np.array([self.earthModel.GetEarthDensityInCGS(p) for p in position])

    def GetPNERatio(self, position):
        proton = int(LeptonInjector.Particle.ParticleType.PPlus)
        offsets = []
        for p in position:
            current_medium = self.earthModel.GetEarthParam(p)
            offsets.append(self.earthModel.GetPNERatio(current_medium.fMediumType_, proton))
        return np.array(offsets)

    # double GetPNERatio(MediumType med, int id) const;
    # double GetPNERatio(MediumType med, int id) const;
    # density_offset =  GetPNERatio(curMedium.fMediumType_ , 2212);
    # EarthParam curMedium=GetEarthParam(pos);

    def prob_area(self, events):
        events = np.asarray(events)
        length = self.chord_length(events)
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
        isTau = ((self.block["final_type_1"] == LeptonInjector.Particle.ParticleType.TauMinus
            or self.block["final_type_1"] == LeptonInjector.Particle.ParticleType.TauPlus)
            or (self.block["final_type_2"] == LeptonInjector.Particle.ParticleType.TauMinus
                or self.block["final_type_2"] == LeptonInjector.Particle.ParticleType.TauPlus))

        use_electron_density = LeptonInjector.getInteraction(
                    LeptonInjector.Particle.ParticleType(self.block["final_type_1"]),
                    LeptonInjector.Particle.ParticleType(self.block["final_type_2"]))

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

        #atmoEntry, atmoExit = self.GetAtmoPoints(pca, direction)

        #segments = self.GetEarthDensitySegments(first_point, last_point, use_electron_density)
        return first_point, last_point

    def prob_pos(self, events):
        events = np.asarray(events)

        #considered_range = get_considered_range(events)

        energy = events["energy"]
        zenith = events["zenith"]
        azimuth = events["azimuth"]
        x = events["x"]
        y = events["y"]
        z = events["z"]

        isTau = ((self.block["final_type_1"] == LeptonInjector.Particle.ParticleType.TauMinus
            or self.block["final_type_1"] == LeptonInjector.Particle.ParticleType.TauPlus)
            or (self.block["final_type_2"] == LeptonInjector.Particle.ParticleType.TauMinus
                or self.block["final_type_2"] == LeptonInjector.Particle.ParticleType.TauPlus))

        use_electron_density = LeptonInjector.getInteraction(
                    LeptonInjector.Particle.ParticleType(self.block["final_type_1"]),
                    LeptonInjector.Particle.ParticleType(self.block["final_type_2"]))

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
        density = self.GetEarthDensityInCGS(position) * self.GetPNERatio(position)
        return density / totalColumnDepth * 1e2 # Convert cm^-1 to m^-1


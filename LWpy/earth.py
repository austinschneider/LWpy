from .generator import generator
import numpy as np
import functools
import EarthModelService
import LeptonInjector

class earth:
    def __init__(self, earth_model_params=None):
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


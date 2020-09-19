from .generator import generator
import numpy as np
import EarthModelService
import LeptonInjector

class volume_generator(generator):
    def __init__(self, block):
        generator.__init__(self, block)

    def prob_area(self, events):
        events = np.asarray(events)
        length = self.chord_length(events)
        radius = self.block["radius"]
        height = self.block["height"]
        volume = p.pi * radius * radius * height
        p_area = length / volume
        p_area /= 1e4 # Convert from m^-2 to cm^-2
        return p_area

    def prob_pos(self, events):
        length = self.get_chord_length(events)
        # length is in m
        return 1.0/length

    def chord_length(self, events):
        # This function finds the length of a chord passing through (x,y,z) at an angle (zenith, azimuth).
        # It gets the intersections with the cylinder, and then calculates the distance between those intersections
        # Uses the same get-cylinder-intersection algorithm as LeptonInjector

        events = np.asarray(events)

        r = self.block["radius"]
        cz1 = -1*self.block["height"]/2.0
        cz2 = -1*cz1

        zenith = events["zenith"]
        azimuth = events["azimuth"]

        nx = np.cos(azimuth)*np.sin(zenith)
        ny = np.sin(azimuth)*np.sin(zenith)
        nz = np.cos(zenith)

        res = np.zeros(len(events))
        nonzero = ~np.logical_and(nx == 0.0, ny == 0.0)
        res[~nonzero] = self.block["height"]/2.0

        nx = nx[nonzero]
        ny = ny[nonzero]
        nz = nz[nonzero]

        x = events["x"][nonzero]
        y = events["y"][nonzero]
        z = events["z"][nonzero]

        zenith = zenith[nonzero]
        azimuth = azimuth[nonzero]

        nx2 = nx*nx
        ny2 = ny*ny
        nr2 = nx2 + ny2
        n_sum = -(nx*x + ny*y)
        r0_2 = x*x+y*y

        root = np.sqrt(n_sum*n_sum - nr2*(r0_2-r*r))

        sol_1 = (n_sum - root)/nr2
        sol_2 = (n_sum + root)/nr2

        # positions corresponding to above solutions
        x1 = x + nx*sol_1
        y1 = y + ny*sol_1
        z1 = z + nz*sol_1
        x2 = x + nx*sol_2
        y2 = y + ny*sol_2
        z2 = z + nz*sol_2

        # check if the solutions are within the z boundaries
        b1_lower = z1<cz1
        b2_lower = z2<cz1
        b1_upper = z1>cz2
        b2_upper = z2>cz2

        bb_lower = np.logical_or(b1_lower, b2_lower)
        bb_upper = np.logical_or(b1_upper, b2_upper)
        bb = np.logical_or(bb_lower, bb_upper)

        # these are the cyliner intersection points. Tentative solutions
        # x1, y1, z1
        # x2, y2, z2

        # replace solutions with encap intersections if necessary
        encap_mask = bb
        bb_bb_lower = bb_lower[bb]
        bb_bb_upper = bb_upper[bb]
        nr = np.sqrt(nr2)
        r0 = np.sqrt(r0_2)

        t1 = (cz1-z)/nz
        xx = x + nx*t1
        yy = y + ny*t1
        zz = cz1
        x1[b1_lower] = xx[b1_lower]
        y1[b1_lower] = yy[b1_lower]
        z1[b1_lower] = zz[b1_lower]
        x2[b2_lower] = xx[b2_lower]
        y2[b2_lower] = yy[b2_lower]
        z2[b2_lower] = zz[b2_lower]

        t2 = (cz2-z)/nz
        xx = x + nx*t2
        yy = y + ny*t2
        zz = cz2
        x1[b1_upper] = xx[b1_upper]
        y1[b1_upper] = yy[b1_upper]
        z1[b1_upper] = zz[b1_upper]
        x1[b2_upper] = xx[b2_upper]
        y1[b2_upper] = yy[b2_upper]
        z1[b2_upper] = zz[b2_upper]

        dist_sq = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
        res[nonzero] = np.sqrt(dist_sq)

        return res

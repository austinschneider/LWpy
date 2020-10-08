from .generator import generator
import numpy as np
import EarthModelService
import LeptonInjector

class volume_generator(generator):
    def __init__(self, block):
        generator.__init__(self, block)

    def inside_volume(self, events):
        events = np.asarray(events)

        radius = self.block["radius"]
        height = self.block["height"]

        x = events["x"]
        y = events["y"]
        z = events["z"]

        r = np.sqrt(x**2 + y**2)

        return np.logical_and(np.abs(z) <= height/2.0, r < radius)

    def prob_area(self, events):
        events = np.asarray(events)
        inside = self.inside_volume(events)

        res = np.zeros(len(events))

        length = self.chord_length(events[inside])
        radius = self.block["radius"]
        height = self.block["height"]
        volume = np.pi * radius * radius * height
        p_area = length / volume
        p_area /= 1e4 # Convert from m^-2 to cm^-2

        res[inside] = p_area
        return res

    def prob_pos(self, events):
        events = np.asarray(events)
        inside = self.inside_volume(events)

        res = np.zeros(len(events))
        length = self.chord_length(events[inside])
        # length is in m
        res[inside] = 1.0/length
        return res

    def chord_length(self, events):
        first_pos, last_pos = self.get_considered_range(events)
        v = (last_pos - first_pos)
        d = np.array([vv.Magnitude() for vv in v])
        return d

    def get_considered_range(self, events):
        # This function finds the length of a chord passing through (x,y,z) at an angle (zenith, azimuth).
        # It gets the intersections with the cylinder, and then calculates the distance between those intersections
        # Uses the same get-cylinder-intersection algorithm as LeptonInjector

        events = np.asarray(events)

        r = self.block["radius"]
        height = self.block["height"]
        cz1 = -self.block["height"]/2.0
        cz2 = self.block["height"]/2.0

        zenith = events["zenith"]
        azimuth = events["azimuth"]

        nx = np.cos(azimuth)*np.sin(zenith)
        ny = np.sin(azimuth)*np.sin(zenith)
        nz = np.cos(zenith)

        nonzero = ~np.logical_and(nx == 0.0, ny == 0.0)

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
        b1_lower = z1<cz1 # first solution below lower z
        b2_lower = z2<cz1 # second solution below lower z
        b1_upper = z1>cz2 # first solution above upper z
        b2_upper = z2>cz2

        bb_lower = np.logical_or(b1_lower, b2_lower) # either solution below lower z
        bb_upper = np.logical_or(b1_upper, b2_upper) # either solution above upper z
        bb = np.logical_or(bb_lower, bb_upper) # either solution out of bounds

        # these are the cyliner intersection points. Tentative solutions
        # x1, y1, z1
        # x2, y2, z2

        # replace solutions with encap intersections if necessary
        encap_mask = bb
        bb_bb_lower = bb_lower[bb]
        bb_bb_upper = bb_upper[bb]
        nr = np.sqrt(nr2)
        r0 = np.sqrt(r0_2)

        # intersection with lower z cap
        t1 = (cz1-z)/nz
        xx = x + nx*t1
        yy = y + ny*t1
        zz = cz1
        x1[b1_lower] = xx[b1_lower]
        y1[b1_lower] = yy[b1_lower]
        z1[b1_lower] = zz
        x2[b2_lower] = xx[b2_lower]
        y2[b2_lower] = yy[b2_lower]
        z2[b2_lower] = zz

        t2 = (cz2-z)/nz
        xx = x + nx*t2
        yy = y + ny*t2
        zz = cz2
        x1[b1_upper] = xx[b1_upper]
        y1[b1_upper] = yy[b1_upper]
        z1[b1_upper] = zz
        x2[b2_upper] = xx[b2_upper]
        y2[b2_upper] = yy[b2_upper]
        z2[b2_upper] = zz

        on_side_1 = np.abs(np.sqrt(x1**2 + y1**2) - r) < 1e-4
        on_side_2 = np.abs(np.sqrt(x2**2 + y2**2) - r) < 1e-4
        between_caps_1 = np.abs(z1) <= height/2.0
        between_caps_2 = np.abs(z2) <= height/2.0
        on_cap_1 = np.logical_or(np.abs(z1 - cz1) < 1e-4, np.abs(z1 - cz2) < 1e-4)
        on_cap_2 = np.logical_or(np.abs(z2 - cz1) < 1e-4, np.abs(z2 - cz2) < 1e-4)
        on_1 = np.logical_or(np.logical_and(on_side_1, between_caps_1), on_cap_1)
        on_2 = np.logical_or(np.logical_and(on_side_2, between_caps_2), on_cap_2)
        if not np.all(on_1):
            mask = ~on_1
            print("radius:", self.block["radius"], "height:", self.block["height"])
            print(np.sum(mask), "/", len(mask), "bad points:")
            print(np.array(list(zip(x1[mask], y1[mask], z1[mask]))))
            r = np.sqrt(x1**2 + y1**2)
            h = z1
            print(np.array(list(zip(r[mask], h[mask]))))
            assert(np.all(on_1))
        if not np.all(on_2):
            mask = ~on_2
            print("radius:", self.block["radius"], "height:", self.block["height"])
            print(np.sum(mask), "/", len(mask), "bad points:")
            print(np.array(list(zip(x2[mask], y2[mask], z2[mask]))))
            r = np.sqrt(x2**2 + y2**2)
            h = z2
            print(np.array(list(zip(r[mask], h[mask]))))
            assert(np.all(on_2))

        i = np.arange(len(events))
        nonzero_index = i[nonzero]
        zero_index = i[~nonzero]
        index = np.empty(len(events)).astype(int)
        index[nonzero] = np.arange(len(nonzero_index))
        index[~nonzero] = np.arange(len(zero_index))

        x = events["x"]
        y = events["y"]
        z = events["z"]

        res = []
        for i in range(len(events)):
            if nonzero[i]:
                p1 = LeptonInjector.LI_Position(x1[index[i]], y1[index[i]], z1[index[i]])
                p2 = LeptonInjector.LI_Position(x2[index[i]], y2[index[i]], z2[index[i]])
            else:
                p1 = LeptonInjector.LI_Position(x[i], y[i], -np.sign(z[i])*height/2.)
                p2 = LeptonInjector.LI_Position(x[i], y[i], np.sign(z[i])*height/2.)
            res.append((p1, p2))

        res = np.array(res).T
        if res.size == 0:
            return np.array([]), np.array([])
        else:
            return res[0], res[1]

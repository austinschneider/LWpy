class spline_repo:
    splines = dict()
    @staticmethod
    def __getitem__(item):
        if item not in splines:
            try:
                spline = photospline.SplineTable(item)
                splines[item] = spline
            except:
                raise ValueError("Spline cannot be opened")
        return splines[item]

import numpy as np
import photospline

class spline_repo_helper(type):
    splines = dict()
    def __getitem__(cls, item):
        if item not in spline_repo_helper.splines:
            try:
                spline = photospline.SplineTable(item)
                spline_repo_helper.splines[item] = spline
            except:
                print(item)
                raise ValueError("Spline cannot be opened")
        return spline_repo_helper.splines[item]

class spline_repo(object, metaclass=spline_repo_helper):
    pass

def eval_spline(spline, coords, grad=None):
    if np.shape(coords) == 2:
        coords = coords.T
    if grad is None:
        return spline.evaluate_simple(coords, 0)
    else:
        try:
            grad = int(grad)
            assert(grad < len(coords))
            grad = 0x1 << grad
            return spline.evaluate_simple(coords, grad)
        except:
            try:
                grad = list(grad)
                res = 0x0
                for i in range(np.shape(coords)[0]):
                    if i in grad:
                        res |= (0x1 << i)
                return spline.evaluate_simple(coords, res)
            except:
                try:
                    return spline.evaluate_simple(coords, grad)
                except:
                    raise ValueError("Could not interpret grad!")


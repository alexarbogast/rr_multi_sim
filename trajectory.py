import numpy as np
import bisect

class Trajectory:
    def __init__(self, points, v, a_max):
        self._points = points
        self._v = v
        self._a_max = a_max
        self._ti = self._find_transit_times()

    def __call__(self, t):
        return self.get_point(t)

    def _find_transit_times(self):
        # these are approximate time (not including accleration)
        diff = np.diff(self._points, axis=0)
        dist = np.insert(np.linalg.norm(diff, axis=1), 0, 0)
        return np.cumsum(dist / self._v)

    def get_point(self, t):
        t = np.clip(t, self._ti[0], self._ti[-1])

        ind = bisect.bisect_left(self._ti, t)
        ind = 1 if ind == 0 else ind

        Ti = self._ti[ind - 1]
        dir = self._points[ind] - self._points[ind - 1]
        vi = self._v * dir / np.linalg.norm(dir)
        pi = self._points[ind - 1] + vi*(t - Ti)
        return pi, vi


class Trajectory5PtP(Trajectory):
    def __init__(self, points, v_approx):
        super(Trajectory5PtP, self).__init__(points, v_approx, 0)
        self._params = self._time_parameterize()

    def _quintic_params(self, theta1, theta2):
        t1, s1, v1, a1 = theta1
        t2, s2, v2, a2 = theta2

        A = np.array([[1, t1, t1**2,   t1**3,    t1**4,    t1**5],
                      [0,  1,  2*t1, 3*t1**2,  4*t1**3,  5*t1**4],
                      [0,  0,     2,    6*t1, 12*t1**2, 20*t1**3],
                      [1, t2, t2**2,   t2**3,    t2**4,    t2**5],
                      [0,  1,  2*t2, 3*t2**2,  4*t2**3,  5*t2**4],
                      [0,  0,     2,    6*t2, 12*t2**2, 20*t2**3]])

        x = np.array([s1, v1, a1, s2, v2, a2])
        b = np.linalg.inv(A) @ x
        return b

    def _time_parameterize(self):
        # elapsed time between points
        elapsed = np.diff(self._ti)
        params = []
        for i in range(0, len(elapsed)):
            theta1 = (0, 0, 0, 0)
            theta2 = (elapsed[i], 1, 0, 0)
            params.append(self._quintic_params(theta1, theta2))
        return params

    def _s(self, b, T):
        return b[5]*T**5 + b[4]*T**4 + b[3]*T**3 + b[2]*T**2 + b[1]*T + b[0]

    def _sd(self, b, T):
        return 5*b[5]*T**4 + 4*b[4]*T**3 + 3*b[3]*T**2 + 2*b[2]*T + b[1]

    def get_point(self, t):
        t = np.clip(t, self._ti[0], self._ti[-1])

        # index of end point
        ind = bisect.bisect_left(self._ti, t) 
        ind = 1 if ind == 0 else ind

        ti = t - self._ti[ind - 1]
        ps = self._points[ind - 1]
        pe = self._points[ind]

        dir = pe - ps
        dir = dir / np.linalg.norm(dir)

        bi = self._params[ind - 1]
        pi = ps + dir*self._s(bi, ti)
        vi = dir * self._sd(bi, ti)
        return pi, vi
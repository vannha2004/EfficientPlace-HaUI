from math import sqrt
import numpy as np


class Accelerate(object):
    def __init__(self, acc: bool = False):
        self.iter = 0
        self.Acc = acc
        self.zu = np.array([])
        self.zv = np.array([])
        self.zz = np.array([])
        self.F = np.array([0.0])
        self.t = (1 + sqrt(5)) / 2.0

    def push(self, uk=None, ukplus=None, vk=None, vkplus=None, zk=None, zkplus=None, nbVar=1):
        if self.Acc:
            tkplus = (1 + sqrt(1 + 4 * self.t * self.t)) / 2.0
            cof = (self.t - 1.0) / tkplus
            self.t = tkplus
            if uk is not None and ukplus is not None:
                self.zu = ukplus + cof * (ukplus - uk)
            if vk is not None and vkplus is not None:
                self.zv = vkplus + cof * (vkplus - vk)
            if zk is not None and zkplus is not None:
                self.zz = zkplus + cof * (zkplus - zk)

    def record_fallback(self, F_val: float):
        """Record function value when acceleration candidate is rejected."""
        self.F = np.append(self.F, float(F_val))

    def acceptz(self, Fz: float, q: int = 3) -> bool:
        if self.Acc:
            if len(self.F) == 1:
                self.F = np.append(self.F, float(Fz))
                return True
            q_eff = min(q, len(self.F))
            accept = False
            for i in range(q_eff):
                if Fz <= self.F[len(self.F) - i - 1]:
                    accept = True
                    break
            if accept:
                self.F = np.append(self.F, float(Fz))
            return accept
        return False

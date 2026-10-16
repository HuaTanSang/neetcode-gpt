import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float,
    ) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = self.sigmoid(z)

        dL_dz = (y_hat - y_true) * y_hat * (1 - y_hat)

        dL_dw = dL_dz * x
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)
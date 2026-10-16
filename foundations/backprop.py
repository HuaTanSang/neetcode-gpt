class Solution:
    def sigmoid(self, z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))

        exp_z = np.exp(z)
        return exp_z / (1.0 + exp_z)

    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float,
    ) -> Tuple[NDArray[np.float64], float]:
        y_hat = self.sigmoid(np.dot(x, w) + b)

        dL_dz = (y_hat - y_true) * y_hat * (1.0 - y_hat)
        dL_dw = dL_dz * x

        np.round(dL_dw, decimals=5, out=dL_dw)

        return dL_dw, round(float(dL_dz), 5)
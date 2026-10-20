import numpy as np
from typing import List


class Solution:
    def relu(self, x): 
        return np.maximum(0, x) 
    
    def sigmoid(self, z): 
        return 1.0 / (1.0 + np.exp(-z)) 

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        m = len(y_true)

        z1 = W1 @ x + b1
        a1 = self.relu(z1)
        y_hat = W2 @ a1 + b2

        # MSE loss
        loss = np.mean((y_hat - y_true) ** 2)

        # Gradient theo output
        g2 = (2.0 / m) * (y_hat - y_true)

        # Gradient của lớp thứ hai
        dW2 = np.outer(g2, a1)
        db2 = g2        
          
        # Truyền gradient về hidden và qua ReLU
        u = W2.T @ g2
        g1 = u * (z1 > 0)

        # Gradient của lớp thứ nhất
        dW1 = np.outer(g1, x)
        db1 = g1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }

        

        










import numpy as np


def mixup_samples(x1, y1, x2, y2, alpha):

    weight = np.random.beta(alpha, alpha)
    weight = max(weight, 1 - weight)

    x = x1 * weight + (1 - weight) * x2
    y = y1 * weight + (1 - weight) * y2

    return x, y

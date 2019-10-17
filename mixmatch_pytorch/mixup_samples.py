def mixup_samples(x1, y1, x2, y2, beta):

    weight = beta.sample().item()
    weight = max(weight, 1 - weight)

    x = x1 * weight + (1 - weight) * x2
    y = y1 * weight + (1 - weight) * y2

    return x, y

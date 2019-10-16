import torch
from .sharpen import sharpen
from .tile_adjacent import tile_adjacent


def guess_targets(features, model, output_transform, K, T):

    with torch.no_grad():
        probabilities = output_transform(model(features))
        probabilities = (
            probabilities
            .view(-1, K, *probabilities.shape[1:])
            .mean(dim=1)
        )
        probabilities = sharpen(probabilities, T)

    return tile_adjacent(probabilities, K)

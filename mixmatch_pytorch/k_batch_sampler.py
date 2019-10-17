import torch
from torch.utils.data import Sampler
from .tile_adjacent import tile_adjacent


class KBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, K):

        self.num_samples = len(dataset)
        self.batch_size = batch_size
        self.K = K

    def __iter__(self):

        while True:
            indices = torch.randint(0, self.num_samples, (self.batch_size,))
            yield tile_adjacent(indices, self.K)

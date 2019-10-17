import torch
from .mixmatch_batch import mixmatch_batch
from .get_unlabeled_loader import get_unlabeled_loader


class MixmatchLoader:

    def __init__(
            self, loader_labeled, dataset_unlabeled, model, output_transform,
            K=2, T=0.5, alpha=0.75
        ):

        self.loader_labeled = loader_labeled
        self.loader_unlabeled = get_unlabeled_loader(
            dataset_unlabeled,
            loader_labeled.batch_size,
            K
        )
        self.model = model
        self.output_transform = output_transform
        self.K = K
        self.T = T
        self.beta = torch.distributions.beta.Beta(alpha, alpha)

    def __iter__(self):

        for batch in self.loader_labeled:
            batch_unlabeled = next(self.loader_unlabeled)
            yield mixmatch_batch(
                batch, batch_unlabeled, self.model, self.output_transform,
                self.K, self.T, self.beta
            )

    def __len__(self):
        
        return len(self.loader_labeled)

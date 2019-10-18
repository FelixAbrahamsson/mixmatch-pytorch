from torch.utils.data import DataLoader
from .k_batch_sampler import KBatchSampler


def get_unlabeled_loader(dataset, batch_size, K, num_workers=1):

    return DataLoader(
        dataset,
        batch_sampler=KBatchSampler(dataset, batch_size, K),
        num_workers=num_workers,
    )

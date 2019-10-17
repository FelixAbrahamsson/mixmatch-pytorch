import torch
from .guess_targets import guess_targets
from .mixup_samples import mixup_samples


def mixmatch_batch(
        batch, batch_unlabeled, model, output_transform, K, T, beta
    ):

    features_labeled = batch['features']
    targets_labeled = batch['targets']
    features_unlabeled = batch_unlabeled['features']
    targets_unlabeled = guess_targets(
        features_unlabeled, model, output_transform, K, T
    )

    indices = torch.randperm(len(features_labeled) + len(features_unlabeled))
    features_W = torch.cat((features_labeled, features_unlabeled), dim=0)[indices]
    targets_W = torch.cat((targets_labeled, targets_unlabeled), dim=0)[indices]

    features_X, targets_X = mixup_samples(
        features_labeled,
        targets_labeled,
        features_W[:len(features_labeled)],
        targets_W[:len(features_labeled)],
        beta
    )
    features_U, targets_U = mixup_samples(
        features_unlabeled,
        targets_unlabeled,
        features_W[len(features_labeled):],
        targets_W[len(features_labeled):],
        beta
    )

    return dict(
        features=torch.cat((features_X, features_U), dim=0),
        targets=torch.cat((targets_X, targets_U), dim=0),
    )

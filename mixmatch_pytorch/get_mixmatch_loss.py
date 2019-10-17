import torch.nn as nn


def get_mixmatch_loss(
        criterion_labeled, output_transform, K=2, weight_unlabeled=100.,
        criterion_unlabeled=nn.MSELoss()
    ):

        def loss_function(logits, targets):

            batch_size = len(logits) // (K + 1)
            loss_labeled = criterion_labeled(
                logits[:batch_size], targets[:batch_size]
            )
            loss_unlabeled = criterion_unlabeled(
                output_transform(logits[batch_size:]), targets[batch_size:]
            )
            return loss_labeled + weight_unlabeled * loss_unlabeled

        return loss_function

import torch


def sharpen(probabilities, T):

    if probabilities.ndim == 1:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
            tempered
            / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )

    else:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered

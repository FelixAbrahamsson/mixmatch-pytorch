import torch


def tile_adjacent(tensor, K):
    ''' Examples:
    in: tensor=[0, 1, 2, 3], K=2
    out: [0, 0, 1, 1, 2, 2, 3, 3]

    in: tensor=[
        [1, 2],
        [3, 4]
    ], K=2
    out: [
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4]
    ]
    '''

    return (
        torch.stack(
            tensor
            .repeat(K, *tuple(1 for i in range(tensor.ndim - 1)))
            .split(tensor.size(0), dim=0)
        )
        .transpose(1, 0)
        .contiguous()
        .view(-1, *tensor.shape[1:])
    )

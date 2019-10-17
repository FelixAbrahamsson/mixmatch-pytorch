# mixmatch-pytorch
An implementation of Mixmatch (https://arxiv.org/pdf/1905.02249.pdf) with PyTorch


## Installation
`pip install https://github.com/FelixAbrahamsson/mixmatch-pytorch`

or clone the repository and run

`pip install .`

## Instructions
The package provides a class `mixmatch.MixMatchLoader` that functions as a normal PyTorch DataLoader, as well as a loss function that is constructed from `mixmatch.get_mixmatch_loss`. For example uses, see below.

You must provide a data loader that functions as an iterable yielding dictionaries with keys `'features'` and `'targets'` that hold augmented (!) features and targets for the labeled dataset. A dataset must also be provided for the unlabeled data, that can be wrapped in a PyTorch DataLoader. The dataset must return dictionaries with key `'features'` that hold augmented features.

A model used for guessing targets for unlabeled data must be provided, as well as an output transform that converts the logits to probabilities.

Your targets may be single class or multiclass, though for a multiclass task take care to use one-hot encoding with a float datatype for your targets. If you want to use this package for a regression task, it should work out of the box with a simple change of input hyperparameters (losses etc.). You would also need to set T=1 to remove sharpening.

For a description of the hyperparameters, please refer to the author's article.

## Example use
```python
loader_mixmatch = mixmatch.MixMatchLoader(
    loader_labeled,
    dataset_unlabeled,
    model,
    output_transform=torch.sigmoid,
    K=2,
    T=0.5,
    alpha=0.75
)

criterion = mixmatch.get_mixmatch_loss(
    criterion_labeled=nn.BCEWithLogitsLoss(),
    output_transform=torch.sigmoid,
    K=2,
    weight_unlabeled=100.,
    criterion_unlabeled=nn.MSELoss()
)

for batch in loader_mixmatch:
    logits = model(batch['features'].to(device))
    loss = criterion(logits, batch['targets'])
```

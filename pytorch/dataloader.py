import torch.utils.data.dataset as torch_data


class KFoldDataset():
    def __init__(self, dataset, kfolds):
        split_length = len(dataset) / kfolds
        splits = [split_length] * kfolds

        # Add any remainder samples to the last subset
        remainder_samples = len(dataset) - sum(splits)
        splits[-1] += remainder_samples

        self.data_subsets = torch_data.random_split(dataset, splits)
        self.kfolds = kfolds

    def get_datasets(self, fold):
        assert(fold < self.kfolds)

        # Create a new dataset without the fold dataset (held for validation)
        train_set = [subset for idx, subset in enumerate(self.data_subsets) if idx != fold]
        val_set = self.data_subsets[fold]
        return torch_data.ConcatDataset(train_set), val_set

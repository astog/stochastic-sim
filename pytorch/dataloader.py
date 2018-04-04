from torch.utils.data.dataset import random_split, ConcatDataset


class KFoldDataLoader():
    def __init__(self, kfolds, dataset):
        split_length = len(dataset) / kfolds
        splits = [split_length] * kfolds

        self.data_subsets = random_split(dataset, splits)
        self.kfolds = kfolds

    def get_datasets(self, fold):
        assert(fold < self.kfolds)

        # Create a new dataset without the fold dataset (held for validation)
        train_set = [subset for idx, subset in enumerate(self.data_subsets) if idx != fold]
        val_set = self.data_subsets[fold]
        return ConcatDataset(train_set), val_set

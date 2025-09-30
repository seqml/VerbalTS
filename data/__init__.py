from torch.utils.data import DataLoader

from .data import CustomDataset

datasets = {
    "custom": CustomDataset
}

class GenerationDataset:
    def __init__(self, configs):
        self.configs = configs
        self.dataset = datasets[configs["name"]](**configs)

    def get_loader(self, split, batch_size, shuffle=True, num_workers=1, include_self=False):
        loader = DataLoader(
            dataset=self.dataset.get_split(split, include_self), 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers)
        return loader

    @property
    def num_attr_ops(self):
        return self.dataset.attr_n_ops
from abc import abstractmethod
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @abstractmethod
    def split_validation(self) -> DataLoader:
        raise NotImplementedError

import abc

from torch.utils.data import Dataset

__all__ = ["DatasetBase"]


class DatasetBase(Dataset):
    @abc.abstractproperty
    def dataset_metadata(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, item: int):
        raise NotImplementedError

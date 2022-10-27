from mimetypes import init
from torch.utils.data import Dataset

class MyDatset(Dataset):
    def __init__(self,data_path,data_num = 2000) -> None:
        super().__init__()
        self.data_path=data_path
        self.data_num= data_num

    def __getitem__(self, index):
        return super().__getitem__(index)
    
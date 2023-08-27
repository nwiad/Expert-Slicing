from torch.utils.data import Dataset
import torch

class SentimentTextDataset(Dataset):
    def __init__(self, labels, matrixes):
        assert len(labels) == len(matrixes), "Bad Lengths"
        self.labels = torch.tensor(labels)
        self.matrixes = torch.stack(matrixes, dim=0)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.labels[index], self.matrixes[index]
    
class FakeDataSet(Dataset):
    def __init__(self, length, hidden_size):
        self.length = length
        self.hidden_size = hidden_size

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return torch.randn(self.hidden_size, dtype=torch.float16), torch.randn(self.hidden_size, dtype=torch.float16)
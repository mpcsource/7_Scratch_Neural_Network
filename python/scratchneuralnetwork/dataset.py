from .tensor import Tensor
import csv

class Dataset:

    # Constructor
    # Initialize by loading data
    def __init__(
            self, 
            path: str
            ) -> None:
        
        self.data_path: str = path
        self.data: list[list] = []

        with open(self.data_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            self.fieldnames = reader.fieldnames
            self.data = [list(row.values()) for row in reader]

        self.mean = 0
        self.standard_deviation = 0
        self.normalized = False

    def __repr__(self) -> str:
        return f'Dataset(data_path="{self.data_path}")'
    
    def head(self) -> list[list]:
        return self.data[:5]
    
    def tail(self) -> list[list]:
        return self.data[:-5]

    def normalize(self):
        self.normalized = True

    def unnormalize(self):
        self.normalized = False
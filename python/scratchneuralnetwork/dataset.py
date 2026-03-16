from .tensor import Tensor

class Dataset:

    # Constructor
    # Initialize by loading data
    def __init__(
            self, 
            data_path: str
            ):
        self.data_path = data_path
        self.x = None
        self.y = None
        self.mean = 0
        self.standard_deviation = 0
        self.normalized = False

    def normalize(self):
        ...

    def unnormalize(self):
        ...
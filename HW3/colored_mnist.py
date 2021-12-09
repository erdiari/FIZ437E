import numpy as np

class coloredMNIST(Dataset):
    def __init__(self, data_dir, test=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if test:
            self.data = np.load(data_dir + 'numpy_train_data.npy')
            self.target = np.load(data_dir + 'numpy_train_label.npy')
        else:
            self.data = np.load(data_dir + 'numpy_test_data.npy')
            self.target = np.load(data_dir + 'numpy_test_label.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__ (self, idx):
        data = self.data[idx]
        target = self.target[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(data)
        return data, target

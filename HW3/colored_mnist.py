import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy

class coloredMNIST(Dataset):
    def __init__(self, data_dir, test=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = test

        if self.is_test:
            self.data = from_numpy(np.load(data_dir + 'numpy_test_data.npz')['arr_0'])
            self.target = from_numpy(np.load(data_dir + 'numpy_test_label.npz')['arr_0'])
        else:
            self.data = from_numpy(np.load(data_dir + 'numpy_train_data.npz')['arr_0'])
            self.target = from_numpy(np.load(data_dir + 'numpy_train_label.npz')['arr_0'])

    def __len__(self):
        return len(self.data)

    def __getitem__ (self, idx):
        if self.transform:
            data = self.transform(data)[idx]
        else:
            data = self.data[idx]
        if self.target_transform:
            target = self.transform(data)[idx]
        else:
            target = self.target[idx]
        return data, target

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    cmnist_train = coloredMNIST('HW3/data/colored_MNIST/',test=False)
    cmnist_test = coloredMNIST('HW3/data/colored_MNIST/',test=True)
    print(cmnist_train.__len__())
    plt.imshow(cmnist_test.__getitem__(277)[0])
    plt.show()

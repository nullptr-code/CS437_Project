from torch.utils.data import Dataset
from skimage.io import imread


class BuildingDataset(Dataset):
    def __init__(
        self, image_paths, label_paths, image_transforms=None, label_transforms=None
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path, label_path = self.image_paths[i], self.label_paths[i]
        image = imread(image_path)
        label = imread(label_path)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.label_transforms:
            label = self.label_transforms(label)

        return image, label

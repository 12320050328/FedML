import torch
import fedml
from fedml import FedMLRunner

import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset


class MedMNISTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y = torch.tensor(y).long().squeeze()
        return x, y

    def __len__(self):
        return len(self.dataset)


def load_data(args):
    data_flag = "pathmnist"

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    class_num = len(info["label"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_dataset = DataClass(split="train", transform=transform, download=True)
    test_dataset = DataClass(split="test", transform=transform, download=True)

    train_dataset = MedMNISTWrapper(train_dataset)
    test_dataset = MedMNISTWrapper(test_dataset)

    client_num = args.client_num_in_total

    client_size = len(train_dataset) // client_num
    lengths = [client_size] * client_num
    lengths[-1] += len(train_dataset) - sum(lengths)

    client_datasets = random_split(train_dataset, lengths)

    train_data_local_dict = {}
    train_data_local_num_dict = {}
    test_data_local_dict = {}

    for client_idx in range(client_num):
        train_data_local_dict[client_idx] = DataLoader(
            client_datasets[client_idx],
            batch_size=args.batch_size,
            shuffle=True
        )

        train_data_local_num_dict[client_idx] = len(client_datasets[client_idx])

        test_data_local_dict[client_idx] = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    train_data_global = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_data_global = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    return dataset, class_num


class CNN(torch.nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    args = fedml.init()

    device = fedml.device.get_device(args)

    dataset, output_dim = load_data(args)

    model = CNN(output_dim)

    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
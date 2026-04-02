from torchvision import datasets
from torch.utils.data import DataLoader
from dataset.transforms import train_transform


def get_dataloaders(data_dir, batch_size=32, num_workers=2):

    print(f"📂 Loading dataset from: {data_dir}")

    # ------------------ TRAIN ------------------
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    # ------------------ VALIDATION ------------------
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=train_transform
    )

    # ------------------ TEST ------------------
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=train_transform
    )

    print(f"✅ Train size: {len(train_dataset)}")
    print(f"✅ Val size: {len(val_dataset)}")
    print(f"✅ Test size: {len(test_dataset)}")

    # ------------------ DATALOADERS ------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
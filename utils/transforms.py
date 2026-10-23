"""
ELEC 475: Lab 2 - PyTorch Transform & Augmentation Pipelines
"""
import torchvision.transforms as transforms
 
DATASET_STATS = {
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std":  [0.2470, 0.2435, 0.2616]
    },
    "tiny_imagenet": {
        "mean": [0.4802, 0.4481, 0.3975],
        "std":  [0.2764, 0.2689, 0.2821]
    }
}
 
def get_train_transforms(dataset: str = "cifar10", augment: bool = True, target_size: int = 64):
    """
    Constructs the training transform pipeline.
    """
    stats = DATASET_STATS.get(dataset.lower(), DATASET_STATS["cifar10"])
 
    transform_list = [
        transforms.Resize((target_size, target_size))
    ]
 
    if augment:
        # Random, label-preserving changes applied each time an image is loaded,
        # so the network sees a slightly different version every epoch.
        transform_list.extend([
            # Pad 4 px on each side (mirroring the edge pixels) and crop a random
            # 64x64 window back out: teaches tolerance to small shifts.
            transforms.RandomCrop(target_size, padding=4, padding_mode="reflect"),
            # Mirror the image left-right half the time: a flipped cat is still a cat.
            transforms.RandomHorizontalFlip(p=0.5),
            # Randomly vary brightness, contrast and saturation by up to +/-20%:
            # teaches tolerance to lighting and camera colour differences.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
 
    transform_list.extend([
        transforms.ToTensor(),                                     # PIL image -> float tensor in [0, 1], shape (C, H, W)
        transforms.Normalize(mean=stats["mean"], std=stats["std"]) # per-channel zero mean, unit variance
    ])
 
    return transforms.Compose(transform_list)
 
def get_test_transforms(dataset: str = "cifar10", target_size: int = 64):
    """
    Constructs deterministic evaluation transform pipeline (no random augmentations).
    """
    stats = DATASET_STATS.get(dataset.lower(), DATASET_STATS["cifar10"])
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"])
    ])
 
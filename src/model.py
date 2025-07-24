import torch
import torch.nn as nn

 

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [B, 3, H, W] → [B, 32, H, W] - image H X W = 224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # [B, 32, H/2, W/2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 32, H/2, W/2] → [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # [B, 64, H/4, W/4]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 64, H/4, W/4] → [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # [B, 128, H/8, W/8]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 28 * 28, 256),  # Assumes input image size is 32x32 → adjust if larger
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x
 
    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

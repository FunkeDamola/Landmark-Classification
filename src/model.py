import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.model = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels = 16, kernel_size=3, padding = 1),     # -> 16X224X224
           nn.BatchNorm2d(16),   
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 16X112*112
            
            nn.Conv2d(16, 32, 3, padding=1),     # -> 32 X 112 X 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 X 56 X 56
            
            nn.Conv2d(32, 64, 3, padding=1), # -> 64 X 56 X 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64 X 28 X 28
            
            nn.Conv2d(64, 128, 3, padding=1), #  -> 128 X 28 X 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # -> 128 X 14 X 14
            
            nn.Flatten(),  
            
            ## Dense Layers
            nn.Linear(128 * 14 * 14, 4000),  # Assuming 8000 neurons in the first dense layer
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # nn.Linear(2000, 1052),  # Second dense layer with 2052 neurons
            # nn.BatchNorm1d(2052),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
       
            nn.Linear(4000, 512),  # Second dense layer with 512 neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, num_classes),  # Output Layer
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


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
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

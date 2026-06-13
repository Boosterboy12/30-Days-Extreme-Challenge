# --- IMPORTING THE DEPENDENCIES --- #
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

# ========================================================================================================================== #
# ========================================================================================================================== #


class ANN(nn.Module):
    # Constructor
    def __init__(self):

        # Initializing Parent Class
        super().__init__()

        # Input Layer -> Hidden Layer
        self.layer1 = nn.Linear(2, 3)

        # Hidden Layer -> Output Layer
        self.layer2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ========================================================================================================================== #
# ========================================================================================================================== #

# Creating Model Object
model = ANN()

# Sample Input
sample_data = torch.tensor([25.0, 180.0])

# Forward Pass
prediction = model(sample_data)
print(prediction)

# ========================================================================================================================== #
#                                                          0 END 1                                                           #
# ========================================================================================================================== #

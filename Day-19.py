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

        # Hidden Layer -> Hidden Layer
        self.layer2 = nn.Linear(3, 4)

        # Hidden Layer -> Hidden Layer
        self.layer3 = nn.Linear(4, 8)

        # Hidden Layer -> Output Layer
        self.layer4 = nn.Linear(8, 2)

    # Forward Pass
    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x


# ========================================================================================================================== #
# ========================================================================================================================== #

# Creating Model Object
model = ANN()

# Sample Input
sample_data = torch.tensor([25.0, 180.0])

# Predicting
prediction = model(sample_data)

print(prediction)

# ========================================================================================================================== #
#                                                          0 END 1                                                           #
# ========================================================================================================================== #

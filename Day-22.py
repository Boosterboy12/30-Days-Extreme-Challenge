import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Iske neeche tumhaara baki ka saara code aayega...
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================= #
# STEP 1: REAL DATA INPUTS (Jo Tumne Samjha Tha)
# ========================================================================= #
# Ek sentence hai jisme 3 words hain. Har word ka 4-dimension ka vector hai.
# X_train = [Word_1, Word_2, Word_3]
X_train = torch.randn(1, 3, 4)

# Asli target output (Y): Hum chahte hain ki is sentence ka answer 1 (Positive) aaye
Y_true = torch.tensor([[1.0]])


# ========================================================================= #
# STEP 2: THE ATTENTION ARCHITECTURE
# ========================================================================= #
class SimpleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Q, K, V nikalne ke liye teen ANNs (Linear Layers)
        self.q_linear = nn.Linear(4, 4, bias=False)
        self.k_linear = nn.Linear(4, 4, bias=False)
        self.v_linear = nn.Linear(4, 4, bias=False)
        # Final word predict karne ke liye classifier layer
        self.classifier = nn.Linear(4, 1)

    def forward(self, x):
        # 1. Q, K, V vectors nikale
        Q, K, V = self.q_linear(x), self.k_linear(x), self.v_linear(x)

        # 2. Relation Scores (Alignment Scores) = Q * K
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 3. Softmax lagakar perfect weights (Context Map) banaya
        context_map = F.softmax(scores, dim=-1)

        # 4. Weighted Sum karke Teesra Value (Context Vector C) nikala
        C = torch.matmul(context_map, V)

        # 5. Average pooling karke single vector banaya aur output predict kiya
        pooled = torch.mean(C, dim=1)
        return torch.sigmoid(self.classifier(pooled))


# ========================================================================= #
# STEP 3: MODEL TRAINING & ACCURACY LOGIC (Sirf 1 Baar)
# ========================================================================= #
model = SimpleAttention()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Weights sudharne wala engine
criterion = nn.BCELoss()  # Galti (Loss) nikalne wala formula

# 1. Forward Pass: Model ne tuka maara (Prediction nikalari)
Y_pred = model(X_train)
print(f"1. Model Ka Guess (Prediction): {Y_pred.item():.4f}")

# 2. Loss Calculation: Galti calculate hui (Y_true aur Y_pred ke beech ka farq)
loss = criterion(Y_pred, Y_true)
print(f"2. Model Ki Galti (Loss Score): {loss.item():.4f}")

# 3. Backward Pass: PyTorch ne dimaag lagakar weights ko sudhara
loss.backward()  # Galti ko piche bheja
optimizer.step()  # Saare weights update kar diye takki agli baar galti kam ho

# 4. Accuracy Check: Kya guess sahi tha?
accuracy = 100.0 if (Y_pred > 0.5).float() == Y_true else 0.0
print(f"4. Is Single Prediction Ki Accuracy: {accuracy}%")

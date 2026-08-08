from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device : {torch.cuda.get_device_name(device)}")
df = pd.read_csv("spambase_csv.csv")
x = df.drop("class", axis=1)
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32, device=device)
train_ds = TensorDataset(x_train_tensor, y_train_tensor)
test_ds = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

class ClassficModel(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

epochs = 100
train_losses = []
eval_losses = []
model = ClassficModel(x_train_tensor.shape[1])
cirtersion = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=.001)

for epoch in range(epochs):
    model.train()
    train_loss_total = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        y_pred = model(batch_x)
        loss = cirtersion(y_pred, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss_total += loss
    train_loss_total /= len(train_loader)
    train_losses.append(train_loss_total)
    model.eval()
    test_loss_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = cirtersion(y_pred, batch_y)
            test_loss_total += loss
        test_loss_total /= len(test_loader)
        eval_losses.append(test_loss_total)
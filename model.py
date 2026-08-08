from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

class ClassficModel(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

epochs = 130
train_losses = []
eval_losses = []
model = ClassficModel(x_train_tensor.shape[1])
model.to(device)
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
        train_loss_total += loss.item()
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
            test_loss_total += loss.item()
        test_loss_total /= len(test_loader)
        eval_losses.append(test_loss_total)
    if epoch % 10 == 0:
        print(f"epoch : {epoch}, train loss : {train_loss_total}, test loss : {test_loss_total}")
model.eval()
with torch.no_grad():
    test_outputs = model(x_test_tensor)  # raw logits
    predictions = torch.sigmoid(test_outputs)  # probabilities 0-1
    predicted_labels = (predictions >= 0.5).float()

    correct = (predicted_labels == y_test_tensor).float().sum()
    accuracy = (correct / len(y_test_tensor)) * 100
    print(f"Correct : {correct}")
    print(f"accuracy : {accuracy}")

    # --- additional metrics beyond accuracy ---
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        classification_report
    )

    y_true_np = y_test_tensor.cpu().numpy().flatten()
    y_pred_np = predicted_labels.cpu().numpy().flatten()
    y_prob_np = predictions.cpu().numpy().flatten()

    cm = confusion_matrix(y_true_np, y_pred_np)
    print(f"\nConfusion matrix (rows=true, cols=predicted):\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    print(f"  True negatives:  {tn}   False positives: {fp}")
    print(f"  False negatives: {fn}   True positives:  {tp}")

    precision = precision_score(y_true_np, y_pred_np)
    recall = recall_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)
    roc_auc = roc_auc_score(y_true_np, y_prob_np)
    pr_auc = average_precision_score(y_true_np, y_prob_np)

    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print(f"\n{classification_report(y_true_np, y_pred_np, target_names=['not spam', 'spam'])}")

    # calibration sanity check: how confident are correct vs incorrect predictions?
    correct_mask = (predicted_labels == y_test_tensor).cpu().numpy().flatten()
    print(f"Avg confidence on correct predictions:   {y_prob_np[correct_mask].mean() if correct_mask.any() else float('nan'):.4f} "
          f"(should be close to 0 or 1)")
    print(f"Avg confidence on incorrect predictions: {y_prob_np[~correct_mask].mean() if (~correct_mask).any() else float('nan'):.4f} "
          f"(closer to 0.5 = model is at least 'unsure' when wrong, which is healthier than confidently wrong)")
plt.plot(range(epochs), train_losses, label="train")
plt.plot(range(epochs), eval_losses, label="test")
plt.title("train and test losses")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
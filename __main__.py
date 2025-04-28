import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def readInfo(filename):
    df = pd.read_csv(filename)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    genre_names = le.classes_
    X = df.iloc[:, 1:-1].copy().to_numpy()
    y = df['label'].copy().to_numpy()
    return X, y, genre_names


class MusicData(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)

    def to_numpy(self):
        return np.array(self.X.view(-1)), np.array(self.y)


class GenreFit(nn.Module):
    def __init__(self):
        super(GenreFit, self).__init__()
        self.norm = nn.BatchNorm1d(58)  # Normalizes
        self.in_to_h1 = nn.Linear(58, 124)
        self.h1_to_h2 = nn.Linear(124, 20)
        self.h2_to_h3 = nn.Linear(20, 73)
        self.h3_to_h4 = nn.Linear(73, 10)

    def forward(self, x):
        x = self.norm(x)  # normalizes
        x = F.sigmoid(self.in_to_h1(x))
        self.dropout = nn.Dropout(0.1)
        x = F.sigmoid(self.h1_to_h2(x))
        x = F.sigmoid(self.h2_to_h3(x))
        return self.h3_to_h4(x)


def trainNN(epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    # load dataset
    # https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data
    X, y, genre_names = readInfo("features_3_sec.csv")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_data = MusicData(X_train, y_train)
    test_data = MusicData(X_test, y_test)

    # create data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    model = GenreFit().to(device)
    print(f"Total parameters: {sum(param.numel() for param in model.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

        # Evaluation
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        preds = torch.argmax(model(x_test), dim=1)
        acc = (preds == y_test_tensor).float().mean()
        print(f"\nTest Accuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test_tensor.cpu(), preds.cpu())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
        disp.plot(xticks_rotation=45)
        plt.title("GTZAN Genre Classification - Confusion Matrix")
        plt.tight_layout()
        plt.show()


trainNN(epochs=10, display_test_acc=True)

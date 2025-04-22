import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from tqdm import tqdm


class MusicData(Dataset):
    def __init__(self):
        # https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data
        df = pd.read_csv("features_3_sec.csv")  # Load our dataset

        # Feature and label data
        self.X = torch.tensor(df.iloc[:, 1:-1:], dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1], dtype=torch.float32)
        # blues = 0, classical=1, country =2, disco=3, hiphop=4, jazz=5, metal=6, pop=7, reggae=8, rock=9

        # Determine the length of the dataset
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X.view(-1)), np.array(self.y)


class GenreFit(nn.Module):
    def __init__(self):
        super(GenreFit, self).__init__()

        self.in_to_h1 = nn.Linear(1, 4)
        self.h1_to_out = nn.Linear(4, 1)

    def forward(self, x):
        x = F.sigmoid(self.in_to_h1(x))
        x = self.h1_to_out(x)
        return x


def trainNN(epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    # load dataset
    data = MusicData()

    # create data loader
    mnist_loader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    number_classify = GenreFit().to(device)
    print(f"Total parameters: {sum(param.numel() for param in number_classify.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(number_classify.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(mnist_loader)):
            x, y = data

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = number_classify(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                predictions = torch.argmax(number_classify(data.test_numbers.to(device)), dim=1)  # Get the prediction
                correct = (predictions == data.test_labels.to(device)).sum().item()
                print(f"Accuracy on test set: {correct / len(data.test_labels):.4f}")


trainNN(epochs=5, display_test_acc=True)

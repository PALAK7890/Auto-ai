import torch
import torch.nn as nn
import torch.optim as optim


class TrainingAgent:

    def train(
        self,
        model,
        X_train,
        y_train,
        task,
        epochs=50
    ):

        X_train = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        if task == "classification":

            y_train = torch.tensor(
                y_train,
                dtype=torch.long
            )

            criterion = nn.CrossEntropyLoss()

        else:

            y_train = torch.tensor(
                y_train,
                dtype=torch.float32
            ).view(-1, 1)

            criterion = nn.MSELoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001
        )

        for epoch in range(epochs):

            optimizer.zero_grad()

            outputs = model(X_train)

            loss = criterion(
                outputs,
                y_train
            )

            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:

                print(
                    f"Epoch {epoch} | Loss {loss.item():.4f}"
                )

        return model
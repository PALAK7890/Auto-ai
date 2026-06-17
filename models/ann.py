import torch.nn as nn


class DynamicANN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_layers,
        output_dim
    ):
        super().__init__()

        layers = []

        prev_dim = input_dim

        for hidden_dim in hidden_layers:

            layers.append(
                nn.Linear(
                    prev_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(0.2)
            )

            prev_dim = hidden_dim

        layers.append(
            nn.Linear(
                prev_dim,
                output_dim
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_infection_classes, num_organ_classes):
        super(MultiTaskModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.shared_fc = nn.Linear(hidden_size, hidden_size)
        self.fc_infection = nn.Linear(hidden_size, num_infection_classes)
        self.fc_organ = nn.Linear(hidden_size, num_organ_classes)
        self.fc_sepsis = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        shared_features = torch.relu(self.shared_fc(lstm_out[:, -1, :]))
        infection_out = torch.softmax(self.fc_infection(shared_features), dim=1)
        organ_out = torch.softmax(self.fc_organ(shared_features), dim=1)
        sepsis_out = torch.sigmoid(self.fc_sepsis(shared_features))
        return infection_out, organ_out, sepsis_out

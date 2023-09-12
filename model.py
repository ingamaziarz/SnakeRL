import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pt'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self, file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        def tensorize(arr, dtype):
            return torch.tensor(arr, dtype=dtype).unsqueeze(0) if len(torch.tensor(arr, dtype=dtype).shape) == 1 else torch.tensor(arr, dtype=dtype)

        state = tensorize(state, torch.float)
        next_state = tensorize(next_state, torch.float)
        action = tensorize(action, torch.long)
        reward = tensorize(reward, torch.float)
        done = tensorize(done, torch.float)

        preds = self.model(state)

        with torch.no_grad():
            targets = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0] * done

        self.optimizer.zero_grad()
        loss = self.criterion(torch.take(preds, action), targets)
        loss.backward()

        self.optimizer.step()


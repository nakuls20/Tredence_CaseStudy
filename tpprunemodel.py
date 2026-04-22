import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 1.0) 

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

class USDSelfPruningNN(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=10):
        super(USDSelfPruningNN, self).__init__()
        self.fc1 = PrunableLinear(input_dim, hidden_dim)
        self.fc2 = PrunableLinear(hidden_dim, hidden_dim // 2)
        self.fc3 = PrunableLinear(hidden_dim // 2, output_dim)
        
        self.erk_scales = self._calculate_erk_scales()

    def _calculate_erk_scales(self):
        layer_params = []
        total_params = 0
        layers = [self.fc1, self.fc2, self.fc3]
        
        for layer in layers:
            p = layer.weight.numel()
            layer_params.append(p)
            total_params += p
        
        scales = [1 - (p / total_params) for p in layer_params]
        return dict(zip(['fc1', 'fc2', 'fc3'], scales))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weighted_sparsity_loss(self):
        total_loss = 0
        total_loss += torch.sum(self.fc1.get_gates()) * self.erk_scales['fc1']
        total_loss += torch.sum(self.fc2.get_gates()) * self.erk_scales['fc2']
        total_loss += torch.sum(self.fc3.get_gates()) * self.erk_scales['fc3']
        return total_loss

    def get_sparsity_level(self, threshold=1e-2):
        total_w, pruned_w = 0, 0
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates = layer.get_gates().detach().cpu().numpy()
            total_w += gates.size
            pruned_w += np.sum(gates < threshold)
        return (pruned_w / total_w) * 100

    def get_all_gate_values(self):
        """Helper to collect all sigmoid gate values into a single array"""
        all_gates = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates = layer.get_gates().detach().cpu().numpy().flatten()
            all_gates.extend(gates)
        return np.array(all_gates)

def train_usd_style(target_lambda=1e-3, epochs=50, device='cpu'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, download=True, transform=transform), batch_size=64)

    model = USDSelfPruningNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"ERK Scales: {model.erk_scales}")

    for epoch in range(epochs):
        if epoch < 10:
            current_lambda = 0.0
        elif epoch < 30:
            current_lambda = target_lambda * (epoch - 10) / 20
        else:
            current_lambda = target_lambda

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            class_loss = criterion(outputs, labels)
            
            sparsity_loss = model.get_weighted_sparsity_loss()
            total_loss = class_loss + current_lambda * sparsity_loss
            
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            sparsity = model.get_sparsity_level()
            print(f"Epoch {epoch+1} | Lambda: {current_lambda:.2e} | Sparsity: {sparsity:.2f}%")

    # Final Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    final_sparsity = model.get_sparsity_level()
    print(f"\nFinal Results -> Accuracy: {100*correct/total:.2f}% | Sparsity: {final_sparsity:.2f}%")

    # --- PLOTTING GATE DISTRIBUTION ---
    gate_values = model.get_all_gate_values()
    
    plt.figure(figsize=(10, 6))
    plt.hist(gate_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Gate Values (Lambda={target_lambda}, Sparsity={final_sparsity:.1f}%)")
    plt.xlabel("Gate Value (Sigmoid Output)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adding vertical line to show typical pruning threshold
    plt.axvline(x=0.01, color='red', linestyle='--', label='Pruning Threshold (0.01)')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_usd_style(target_lambda=1e-3, epochs=50, device=device)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
import cv2
import os

# Dataset personalizado para capturar frames do vídeo
class VideoFrameDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.frames = self._load_frames()
        self.transform = transform

    def _load_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

# Rede Neural Simples (Baseada no ResNet-18)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 2 classes: normal ou anomalia

    def forward(self, x):
        return self.resnet(x)

# Função de treinamento
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs.to(device)

            # Zerar gradientes
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            labels = torch.zeros(inputs.size(0), dtype=torch.long).to(device)  # Todas como 'normal' inicialmente
            loss = criterion(outputs, labels)

            # Backward e otimização
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader.dataset)}")

# Configurações
video_path = 'video_treino.mp4'
batch_size = 16
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pré-processamento de imagens
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset e DataLoader
dataset = VideoFrameDataset(video_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo, critério de perda e otimizador
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs)

# Salvando o modelo
torch.save(model.state_dict(), 'modelo_treinado.pth')
print("Treinamento completo e modelo salvo.")

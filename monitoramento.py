import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Rede Neural Simples (Baseada no ResNet-18)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 2 classes: normal ou anomalia

    def forward(self, x):
        return self.resnet(x)

# Função de monitoramento em tempo real
def monitor_process(model, transform):
    cap = cv2.VideoCapture(0)  # Usa a câmera padrão

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pré-processar o frame
        img_tensor = transform(frame).unsqueeze(0).to(device)

        # Fazer inferência
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)

        # Verificar se há anomalia
        if preds.item() == 1:
            cv2.putText(frame, 'Anomalia Detectada!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Processo Normal', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar o frame com a detecção
        cv2.imshow('Monitoramento', frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo treinado
model = SimpleModel().to(device)
model.load_state_dict(torch.load('modelo_treinado.pth'))
model.eval()

# Pré-processamento de imagens
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Iniciar monitoramento
monitor_process(model, transform)

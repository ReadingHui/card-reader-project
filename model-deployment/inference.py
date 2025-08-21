import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the same model structure
class ResNet18WithDropout(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

def load_model(path='model.pth'):
    base = models.resnet18(pretrained=True)
    model = ResNet18WithDropout(base, 165)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image, model, transform):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
    return class_idx
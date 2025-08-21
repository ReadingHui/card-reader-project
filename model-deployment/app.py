import gradio as gr
from inference import load_model, predict
from torchvision import transforms
import json

model = load_model('resnet18_v1.pth')

transform = transforms.Compose([
    transforms.Resize((342, 245)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

def classify_image(img):
    class_idx = predict(img, model, transform)
    return idx_to_class[class_idx]

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(),
    title="Image Classifier",
    description="Upload an image to classify."
).launch()

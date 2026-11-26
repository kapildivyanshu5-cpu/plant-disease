import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class Tomato(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)


@st.cache_resource
def load_model():
    with open("class_names.json", "r") as file:
        class_names = json.load(file)

    model = Tomato(num_classes=len(class_names))
    model.load_state_dict(torch.load("best_improved.pt", map_location="cpu"))
    model.eval()
    return model, class_names


model, class_names = load_model()
st.write("Model loaded, classes:", class_names)

def clean_name(raw_name):
    return (
        raw_name.replace("Tomato__Tomato_", "")
        .replace("Tomato__", "")
        .replace("Tomato_", "")
        .replace("__", " ")
        .replace("_", " ")
    )


eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("Tomato Leaf Disease Classifier")

uploaded_file = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = eval_transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, 3)
    top_names = [clean_name(class_names[index]) for index in top_indices.tolist()]
    top_values = [value * 100 for value in top_probabilities.tolist()]

    # Grad-CAM (runs outside no_grad because it needs gradients)
    rgb_image = np.array(image.resize((128, 128))).astype(np.float32) / 255.0
    with GradCAM(model=model, target_layers=[model.bn3]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    heatmap_overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    st.subheader(f"{top_names[0]} ({top_values[0]:.1f}%)")

    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image.resize((128, 128)), caption="Uploaded leaf", use_container_width=True)
    with right_column:
        st.image(heatmap_overlay, caption="Grad-CAM: where the model looked", use_container_width=True)

    chart_data = pd.DataFrame({"Confidence (%)": top_values}, index=top_names)
    st.bar_chart(chart_data)
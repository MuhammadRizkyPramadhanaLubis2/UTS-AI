import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. SETUP DEVICE & TRANSFORMS
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

val_test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Cat', 'Dog', 'Wild'] 

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding= 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding= 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding= 1)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)

        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)

        return x

# ==========================================
# 3. LOAD MODEL & PREDICT FUNCTION
# ==========================================
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("animal_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

def predict_image(image, model):
    image = image.convert('RGB')
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output, axis=1).item()
        
    return class_names[predicted_idx]

# ==========================================
# 4. WEBSITE INTERFACE (STREAMLIT UI)
# ==========================================
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

st.title("🐾 Animal Face Classifier")
st.write("This app uses PyTorch to guess the type of animal from your uploaded image.")
st.write("Supported categories: **Cat, Dog, and Wild**.")

uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    st.write("Analyzing image...")
    
    try:
        label = predict_image(image, model)
        st.success(f"Model Prediction: It's a **{label}**!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Make sure the 'animal_classifier_model.pth' file is in the same folder as app.py.")
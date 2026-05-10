# Load model from Registry and run Inference

import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms

MODEL_NAME = "CrashBarrier_ResNet18"
MODEL_STAGE = "champion"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

model_uri = "models:/CrashBarrier_ResNet18@champion"

model = mlflow.pytorch.load_model(model_uri)

model = model.to(DEVICE)

model.eval()

print("Loaded model from registry:")
print(model_uri)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = r"C:\Users\Admin\capstone_proj\3b997b63-2026May04-142508399_0048.png"

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

prediction = prediction.item()

confidence = confidence.item()

classes = {
    0: "Good",
    1: "Damaged"
}

predicted_class = classes[prediction]

print("\nPrediction Results")
print("-------------------------")
print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.4f}")
import torch
import torch.nn as nn
import cv2 as cv
from torchvision import models, transforms
from PIL import Image

# 1. Define the classes
classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(classes)

# 2. Re-create the Architecture -> identical to the trained model
def load_trained_model(weights_path):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.mobilenet_v3_small(weights=None)
    # Custom classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # Load the local file
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() # set to evaluation mode
    return model, device

# 3. Set up the preprocessing -> transforms, the same from val_transform
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 4. prediction function
def predict(frame, model, device):
    # Convert BGR (OpenCV) to RGB(PIL/Torch)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #Apply transforms and move to device
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return classes[predicted.item()]

# -------------------------------------
# Returning: Recyling, Waste, or Reject
# -------------------------------------
def waste_classfication_decision(result):

    recycle = ['cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes']
    waste = ['biological', 'trash']
    
    if result.lower() in recycle: return "RECYCLABLE"
    if result.lower() in waste: return "WASTE"
    if result.lower() == 'battery': return "REJECT (BATTERY)"
    return "UNKNOWN"

# --------------------------------
# Cellphone Camera
# --------------------------------
model, device = load_trained_model("/Users/guicataneo/Desktop/Bio Project/ML Model/waste_predict_model.pth")

cap = cv.VideoCapture(0)
print("Starting live feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found...")
        break
    
    # ------------------
    # Execution
    # ------------------
    category = predict(frame, model, device)
    decision = waste_classfication_decision(category)

    display_text = f"{category.upper()} -> {decision}"
    cv.putText(frame, display_text, (30, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

    cv.imshow('Waste Classifirer AI', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Define the classes
classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(classes)

# 2. Re-create the Architecture -> identical to the trained model
def load_trained_model(weights_path):
    model = models.mobilenet_v3_small(weights=None)

    # Custom classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # Load the local file
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval() # set to evaluation mode
    return model

# 3. Set up the preprocessing -> transforms, the same from val_transform
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 4. prediction function
def predict(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image.show()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return classes[predicted.item()]

# -------------------------------------
# Returning: Recyling, Waste, or reject
# -------------------------------------
def waste_classfication_decision(result):

    recycle = ['cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes']
    waste = ['biological', 'trash']
    reject = 'battery'

    # 1 -> recycle
    # 2-> waste
    # 3-> reject

    answer = 0

    if result.lower() in recycle:
        answer = 1
    
    if result.lower() in waste:
        answer = 2
    
    if result.lower() == reject:
        answer = 3

    return answer
    
# ------------------
# Execution
# ------------------
model = load_trained_model("/Users/guicataneo/Desktop/Bio Project/ML Model/waste_predict_model.pth")
result = predict("/Users/guicataneo/Downloads/plastic_bottle.jpg", model)
print(f"The predicted waste category is: {result}")
answer = waste_classfication_decision(result)


if answer == 1:
    print("This is: Recyclabel")
elif answer == 2:
    print("This is: Waste")
elif answer == 3:
    print("This is: Reject")

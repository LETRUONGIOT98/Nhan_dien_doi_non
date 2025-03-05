import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import numpy as np

# Load the saved model as an OrderedDict
state_dict = torch.load('flower_classification_model.pth', map_location=torch.device('cpu'))

# Extract the model from the OrderedDict
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
model.load_state_dict(state_dict)
model.eval()

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict class given an input image
def predict(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_class = output.max(1)
    return predicted_class.item()

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to a PIL Image
    image = Image.fromarray(frame)
    
    # Perform prediction
    predicted_class = predict(image)
    
    # Map predicted class to class name
    class_names = ['Hoa cuc', 'Bo cong anh']  # Make sure these class names match your training data
    predicted_class_name = class_names[predicted_class]
    
    # Display the frame with predicted class name
    cv2.putText(frame, f'Predicted: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

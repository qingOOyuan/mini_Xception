import cv2
import numpy as np
import torch
from utils.Model import mini_XCEPTION

# Set the device to GPU if available, otherwise, use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

def preprocess_input(image_pixels):
    """
    Preprocess the image for model input

    :param image_pixels: Numpy array of image pixels
    :return: Tensor of preprocessed image pixels
    """
    image_pixels = image_pixels.astype('float32')
    image_pixels = image_pixels / 255.0  # Normalize to [0, 1]
    image_pixels = image_pixels - 0.5    # Center to [-0.5, 0.5]
    image_pixels = image_pixels * 2.0    # Scale to [-1, 1]
    return torch.tensor(image_pixels)

# Load and convert the image to grayscale
original_image = cv2.imread("test.jpeg")
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Paths to the face detection model and the trained emotion detection model
face_detection_model_path = 'utils/haarcascade_frontalface_default.xml'
emotion_detection_model_path = 'output/Epoch_200_emotion.pth'
# Dictionary mapping emotion class indices to human-readable labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Load the Haar cascade for face detection
face_detector = cv2.CascadeClassifier(face_detection_model_path)
# Load the emotion detection model
emotion_detector = mini_XCEPTION(num_classes=7).to(device)
emotion_detector.load_state_dict(torch.load(emotion_detection_model_path, map_location=device))

# Define the input size for the emotion detection model
input_size = (48, 48)

# Detect faces in the image
faces = face_detector.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=8)

with torch.no_grad():  # Inference mode, no gradients needed
    for face_coords in faces:
        x, y, width, height = face_coords
        face = grayscale_image[y:y + height, x:x + width]
        try:
            face = cv2.resize(face, input_size)
        except:
            continue  # Skip the face if resizing fails
        face = preprocess_input(face)
        face_tensor = torch.unsqueeze(face, 0)
        face_tensor = torch.unsqueeze(face_tensor, 0)
        face_tensor = face_tensor.to(device)
        # Predict the emotion class
        predicted_emotion_idx = np.argmax(emotion_detector(face_tensor)).item()
        predicted_emotion_text = emotion_labels[predicted_emotion_idx]

        print("Prediction:", predicted_emotion_text)
        # Draw a rectangle around the detected face
        cv2.rectangle(original_image, (x, y), (x + width, y + height), (0, 0, 255), 1)
        # Display the image with the detected face and prediction
        cv2.imshow("Detected emotion", original_image)
        cv2.waitKey(0)  # Wait for a key press to exit

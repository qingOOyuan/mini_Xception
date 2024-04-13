import cv2
import numpy as np
import torch
from utils.Model import mini_XCEPTION

# Using OpenCV's default Haar feature-based cascade classifier for face detection, which may not be highly accurate

# Choose the device to run the model on (GPU if available, else CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

def preprocess_input(image):
    # Converts the input image to float32, scales the pixel values to [-1, 1]
    image = image.astype('float32')
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    return torch.tensor(image)

# Load image in grayscale mode
input_image = cv2.imread("test.jpeg", 0)

# Path to the face detector model and the emotion recognition model
face_detector_path = 'utils/haarcascade_frontalface_default.xml'
emotion_recognition_model_path = 'output/Epoch_200_emotion.pth'

# Mapping of emotion labels
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# Load the face detector and emotion recognition models
face_detector = cv2.CascadeClassifier(face_detector_path)
emotion_model = mini_XCEPTION(num_classes=7).to(device)
emotion_model.load_state_dict(torch.load(emotion_recognition_model_path, map_location=device))

# Size to resize faces to before feeding into emotion recognition model
emotion_input_size = (48, 48)

# Open a handle to the webcam
camera = cv2.VideoCapture(0)

with torch.no_grad():  # Inference without tracking gradients
    while True:
        successful_frame_read, frame = camera.read()
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, width, height) in detected_faces:
            # Extract the face region from the grayscale image
            face_region = input_image[y:y + height, x:x + width]
            try:
                resized_face = cv2.resize(face_region, emotion_input_size)
            except:
                continue  # Skip the rest of the loop if face can't be resized
            preprocessed_face = preprocess_input(resized_face)
            face_tensor = torch.unsqueeze(preprocessed_face, 0)
            face_tensor = torch.unsqueeze(face_tensor, 0)
            face_tensor = face_tensor.to(device)
            
            # Predict the emotion
            emotion_prediction = np.argmax(emotion_model(face_tensor)).item()
            predicted_emotion = emotion_labels[emotion_prediction]

            print("Predicted emotion:", predicted_emotion)
            
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 1)
            
        # Display the resulting frame
        cv2.imshow("Emotion Detector", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam handle and destroy all OpenCV windows
camera.release()
cv2.destroyAllWindows()

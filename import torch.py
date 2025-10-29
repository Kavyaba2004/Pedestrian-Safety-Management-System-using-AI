import torch
import cv2
import pygame
import numpy as np

# Initialize Pygame for sound and traffic light simulation
pygame.init()
pygame.mixer.init()

# Load the pre-trained YOLOv5 model for pedestrian detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 small model for faster inference

# Sound alert when pedestrian is detected
try:
    sound = pygame.mixer.Sound("sound.mp3")  # Ensure the sound file is in the current directory
except pygame.error as e:
    print(f"Error loading sound: {e}")
    sound = None

# Initialize the camera
cap = cv2.VideoCapture("vid.mp4")  # Use 0 for default webcam, or specify a path for video file
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set up Pygame window for traffic light simulation
screen = pygame.display.set_mode((200, 300))
pygame.display.set_caption("Traffic Light Simulation")

# Function to draw the traffic light
def draw_traffic_light(color):
    screen.fill((255, 255, 255))  # Clear the screen
    if color == "RED":
        pygame.draw.circle(screen, (255, 0, 0), (100, 150), 50)  # Red light
    elif color == "GREEN":
        pygame.draw.circle(screen, (0, 255, 0), (100, 150), 50)  # Green light
    pygame.display.flip()

# Function to control the traffic light based on pedestrian detection
def traffic_light_control(pedestrian_detected):
    if pedestrian_detected:
        draw_traffic_light("RED")  # Turn light red if pedestrian is detected
        if sound:
            sound.play()  # Play sound alert
    else:
        draw_traffic_light("GREEN")  # Turn light green if no pedestrian

# Function to detect pedestrians in the video feed
def detect_pedestrians(frame):
    results = yolo_model(frame)  # Run YOLOv5 on the frame
    pedestrians = results.xyxy[0][results.xyxy[0][:, -1] == 0]  # Filter for "person" class (0)
    return pedestrians

# Main loop to process video frames and control the traffic light
while cap.isOpened():
    ret, frame = cap.read()  # Read the frame from the camera
    if not ret:
        break
    
    # Detect pedestrians
    pedestrians = detect_pedestrians(frame)
    pedestrian_detected = len(pedestrians) > 0  # Check if any pedestrian is detected
    
    # Update the traffic light based on pedestrian detection
    traffic_light_control(pedestrian_detected)

    # Draw bounding boxes around detected pedestrians
    for (x1, y1, x2, y2, conf, cls) in pedestrians:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Display the processed video frame with pedestrian detection
    cv2.imshow('Pedestrian Detection', frame)
    
    # Check for exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()

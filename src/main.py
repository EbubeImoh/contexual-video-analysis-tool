from moviepy.editor import VideoFileClip
import speech_recognition as sr
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions # type: ignore
from keras.layers import Conv3D, MaxPooling3D, Reshape, LSTM, Dense # type: ignore
from keras.models import Sequential # type: ignore
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18  # ResNet3D model
from ultralytics import YOLO
from fer import FER
import face_recognition
from PIL import Image
import numpy as np
import cv2 as cv
import os 

# EXTRACT FRAMES FROM VIDEO FILE

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video = cv.VideoCapture(video_path)

    frame_paths = []
    frame_index = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break  # Break if no more frames are available

        # Define the path to save the frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        cv.imwrite(frame_filename, frame)  # Save the frame as a JPEG file
        frame_paths.append(frame_filename)  # Store the path of the saved frame
        
        frame_index += 1

    video.release()
    
    return frame_paths  # Return the list of frame file paths

# CLASSIFY FRAMES

model = ResNet50(weights='imagenet')

def classify_frame(frame_path):
    img = image.load_img(frame_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]


# MAP FACES TO KNOW FACE ENCODINGS

def recognize_faces_in_frames(frames):
    recognized_faces = {}
    
    for frame_path in frames:
        image = face_recognition.load_image_file(frame_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        recognized_faces[frame_path] = {
            'locations': face_locations,
            'encodings': face_encodings
        }
    
    return recognized_faces

# REVERT BACK TO BUILD CUSTOM ACTION DETECTION MODEL

# def build_action_recognition_model():
#     model = Sequential()
    
#     # 3D Convolution Layer
#     model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(frames, height, width, channels)))
    
#     # 3D MaxPooling Layer
#     model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
#     # Reshape Layer to flatten height and width dimensions
#     model.add(Reshape((-1, 32)))  # Flatten spatial dimensions, keep channels
    
#     # LSTM Layer
#     model.add(LSTM(100, activation='relu'))
    
#     # Dense Layer
#     model.add(Dense(1, activation='sigmoid'))
    
#     return model


# YOU ONLY LOOK ONCE OBJECT DETECTION MODEL 

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# USING PYTORCH TO LOAD THE MODEL

def detect_objects(frames):
    # Load YOLOv5 model from Ultralytics
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    
    detected_objects = {}
    
    for frame_path in frames:
        img = Image.open(frame_path)
        results = model(img)
        detected_objects[frame_path] = results.pandas().xyxy[0].to_dict(orient="records")
    
    return detected_objects

# EMOTION DETECTION 

def detect_emotions(frames):
    detector = FER()  # Initialize FER detector
    detected_emotions = {}
    
    for frame_path in frames:
        # Read the image from the frame path
        img = cv.imread(frame_path)
        
        # Convert image from BGR (OpenCV format) to RGB (FER format)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Detect emotions in the image
        result = detector.detect_emotions(img_rgb)
        
        # Extract the dominant emotion
        if result:
            emotions = result[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
        else:
            dominant_emotion = "unknown"
        
        detected_emotions[frame_path] = dominant_emotion
    
    return detected_emotions

# FOR NOW I AM USING A GENERIC ACTION RECOGNITION MODEL FROM RESTNET3D

def recognize_actions(frames):
    # Load a pre-trained ResNet3D model for action recognition
    model = r3d_18(pretrained=True)
    model.eval()
    
    # Define the transformation to apply to each frame
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Prepare a list to store the transformed frames
    processed_frames = []
    
    for frame_path in frames:
        img = cv.imread(frame_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
        transformed_frame = transform(img)
        processed_frames.append(transformed_frame)
    
    # Stack frames along the depth dimension (to match [C, T, H, W])
    clip = torch.stack(processed_frames, dim=1).unsqueeze(0)  # Shape: [1, C, T, H, W]
    
    # Predict actions in the sequence of frames
    with torch.no_grad():
        outputs = model(clip)
    
    # Get the predicted action
    _, predicted = outputs.max(1)
    action_label = predicted.item()  # Convert tensor to integer
    
    return action_label

# CONTEXT ANALYSIS (PUTTING EVERYTHING TOGETHER)
# REMINDER --- APPLY GENERATIVE AI TO IMPROVE RESULTS (MAY SLOW DOWN RUNNING TIME THOUGH)

def analyze_context(detected_objects, recognized_faces, detected_emotions, actions, transcribed_text):
    # Generate a text-based summary
    
    # Extract object names from detected_objects
    object_names = []
    for objs in detected_objects.values():
        for obj in objs:
            if isinstance(obj, dict):
                object_names.append(obj.get('name', 'Unknown Object'))  # Handling if 'name' key is missing
            else:
                object_names.append(obj)  # In case obj is a string directly

    summary = f"Actions: {actions}\n"
    summary += f"Objects Detected: {', '.join(set(object_names))}\n"
    
    # Extract face names from recognized_faces
    face_names = []
    for faces in recognized_faces.values():
        for face in faces:
            if isinstance(face, dict):
                face_names.append(face.get('name', 'Unknown Face'))  # Handling if 'name' key is missing
            else:
                face_names.append(face)  # In case face is a string directly
    
    summary += f"Faces Recognized: {', '.join(set(face_names))}\n"
    summary += f"Emotions Detected: {', '.join(set(detected_emotions.values()))}\n"
    summary += f"Transcribed Speech: {transcribed_text}\n"
    
    return summary


# EXTRACT AUDIO FROM VIDEO FILE

def extract_audio(video_path, aud_path):
    # Load the video
    video = VideoFileClip(video_path)
    
    # Check if the video has an audio track
    if not video.audio:
        print("No audio track found in the video.")
        return None
    
    # Save the audio as a WAV file
    video.audio.write_audiofile(aud_path, codec='pcm_s16le')
    
    return aud_path

# PROCESS AUDIO USING THE SPEECH RECOGNITION LIBRARY

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    

# IF THE AUDIO HAS A LOT OF NOISE, THIS IS A CLEANING ALGORITHM THAT CAN DENOISE IT
# DENOISING ALGORITHM

# import noisereduce as nr

# def advanced_noise_reduction(audio_array, sample_rate):
#     # Apply noise reduction
#     reduced_noise_audio = nr.reduce_noise(y=audio_array, sr=sample_rate)
#     return reduced_noise_audio

# cleaned_audio = advanced_noise_reduction(audio_array, sample_rate)
# save_audio(cleaned_audio, sample_rate, "cleaned_test_audio_advanced.wav")

def analyze_video(video_path):
    # Step 1: Extract frames and audio
    frames = extract_frames(video_path, 'images/frames')
    audio = extract_audio(video_path, 'audio/test_audio.wav')
    
    # Step 2: Detect objects, faces, emotions in each frame
    detected_objects = detect_objects(frames)
    recognized_faces = recognize_faces_in_frames(frames)
    detected_emotions = detect_emotions(frames)
    
    # Step 3: Recognize actions in the sequence of frames
    actions = recognize_actions(frames)
    
    # Step 4: Transcribe speech in the audio
    transcribed_text = transcribe_audio(audio)
    
    # Step 5: Contextual analysis (Combine all information)
    context = analyze_context(detected_objects, recognized_faces, detected_emotions, actions, transcribed_text)
    
    # Step 6: Output the context
    return context

context_summary = analyze_video('video/test_video.mp4')
print(context_summary)
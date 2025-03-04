#run python -m streamlit run app1.py command in the terminal to run the application
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import os
import streamlit as st
from transformers import ViTConfig, ViTForImageClassification
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import interpolate
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.platypus import Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import io
import tempfile
from datetime import datetime
import base64
import tensorflow as tf
import cv2


st.set_page_config(page_title="ECG Analysis App", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# st.sidebar.write(f"Using device: {device}")

# HuggingFace Configuration
HUGGINGFACE_API_KEY = "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"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize session state to store user data
if 'page' not in st.session_state:
    st.session_state.page = 'user_details'
    
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'name': '',
        'age': '',
        'sex': '',
        'dietary_habits': [],
        'exercise_habits': [],
        'allergies': '',
        'medical_history': [],
    }

def change_page(page):
    st.session_state.page = page

class CNNModel(nn.Module):
    def __init__(self, num_classes, device):
        super(CNNModel, self).__init__()
        self.cnn = models.resnet18(weights=None)  # Don't download pre-trained weights
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, num_classes)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        return self.cnn(x)

# Define ViT Model (without downloading pre-trained weights)
class ViTModel(nn.Module):
    def __init__(self, num_classes, device):
        super(ViTModel, self).__init__()
        # Create a simple feature extractor that won't download anything
        self.feature_extractor = lambda x, return_tensors: {"pixel_values": x}
        # Create a ViT model with the right architecture but no pre-trained weights
        config = ViTConfig(num_labels=num_classes)
        self.model = ViTForImageClassification(config)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        feature_input = self.feature_extractor(x, return_tensors="pt")
        feature_input = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in feature_input.items()}
        logits = self.model(**feature_input).logits
        return logits

# Define Ensemble Model
class EnsembleModel(nn.Module):
    def __init__(self, num_classes, device):
        super(EnsembleModel, self).__init__()
        self.cnn_model = CNNModel(num_classes, device)
        self.vit_model = ViTModel(num_classes, device)
        # This matches the error message: expecting fc.weight with shape [6, 12]
        self.fc = nn.Linear(num_classes * 2, num_classes)  # num_classes * 2 = 12 for 6 classes
        self.device = device

    def forward(self, x):
        # During inference, we'll just pass through without using the sub-models
        # This is because we'll load all weights from the state dict directly
        return self.fc(torch.zeros(x.size(0), self.fc.in_features, device=self.device))

def preprocess_ecg_signal(signal_data):
    """
    Preprocess ECG signal according to the methodology:
    1) Split continuous ECG signal to 10s windows
    2) Normalize amplitude to [0, 1]
    3) Find local maximums
    4) Find R-peak candidates (threshold 0.9)
    5) Find median R-R interval as nominal heartbeat period (T)
    6) Extract beats with length 1.2T centered on each R-peak
    7) Pad to fixed length
    
    Returns extracted beats as images
    """
    # Normalize the signal to [0, 1]
    normalized_signal = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
    
    # Find all local maximums (using scipy.signal.find_peaks)
    peaks, _ = find_peaks(normalized_signal, height=0.9)  # R-peak candidates with threshold 0.9
    
    if len(peaks) < 2:
        st.warning("Not enough R-peaks detected. Try a different signal or threshold.")
        return None
    
    # Calculate R-R intervals and find median as nominal heartbeat period (T)
    rr_intervals = np.diff(peaks)
    T = int(np.median(rr_intervals))
    
    # Extract beats of length 1.2T
    beat_length = int(1.2 * T)
    fixed_length = 224  # Fixed length for model input
    
    extracted_beats = []
    beat_images = []
    
    for peak in peaks:
        # Ensure we have enough signal before and after the peak
        start = max(0, peak - beat_length // 2)
        end = min(len(normalized_signal), peak + beat_length // 2)
        
        if end - start < beat_length // 2:
            continue  # Skip if we can't get a full beat
            
        # Extract the beat
        beat = normalized_signal[start:end]
        
        # Pad or resize to fixed length
        if len(beat) < fixed_length:
            # Pad with zeros
            padded_beat = np.zeros(fixed_length)
            padded_beat[:len(beat)] = beat
            beat = padded_beat
        else:
            # Resize to fixed length
            x_old = np.linspace(0, 1, len(beat))
            x_new = np.linspace(0, 1, fixed_length)
            f = interpolate.interp1d(x_old, beat)
            beat = f(x_new)
            
        extracted_beats.append(beat)
        
        # Convert 1D beat to 2D image (224x224)
        beat_image = create_2d_ecg_image(beat)
        beat_images.append(beat_image)
    
    return extracted_beats, beat_images

def create_2d_ecg_image(beat_data):
    """
    Convert 1D ECG beat to 2D image suitable for CNN input
    """
    # Create a figure without a visible frame
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.plot(beat_data, color='black', linewidth=2)
    ax.axis('off')
    ax.set_ylim(0, 1)
    
    # Convert plot to image
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Convert to PIL Image
    ecg_image = Image.fromarray(image_from_plot)
    return ecg_image

# Function to detect whether the uploaded file is a raw ECG signal or processed beat image
def detect_file_type(uploaded_file):
    try:
        # Try to open as image
        Image.open(uploaded_file)
        return "image"
    except:
        # If not an image, try to read as numerical data
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try to read file as CSV or text with numerical data
            try:
                data = np.loadtxt(uploaded_file, delimiter=',')
            except:
                uploaded_file.seek(0)
                data = np.loadtxt(uploaded_file, delimiter='\t')
                
            return "ecg_signal"
        except:
            return "unknown"

# Preprocessing Function for images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Convert to RGB in case the image is RGBA or grayscale
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

# Load Class Mapping
@st.cache_resource
def load_class_mapping(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading class mapping: {e}")
        return None

# Load Model
@st.cache_resource
def load_model(model_path, num_classes):
    try:
        # Create a model with the correct architecture
        model = EnsembleModel(num_classes, device)
        
        # Load the state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# This is the actual inference function that will be used
def ensemble_inference(model, x):
    x = x.to(model.device)
    cnn_output = model.cnn_model(x)
    vit_output = model.vit_model(x)
    combined_output = torch.cat((cnn_output, vit_output), dim=1)
    return model.fc(combined_output)

# Perform Prediction
def predict_class(model, image, idx_to_class):
    with torch.no_grad():
        try:
            # Use our custom inference function
            outputs = ensemble_inference(model, image)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_idx = predicted_idx.item()
            
            condition = idx_to_class.get(str(predicted_idx), f"Unknown class: {predicted_idx}")
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted_idx].item() * 100
            return condition, confidence, probabilities.cpu().numpy()
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return "Error", 0, np.zeros(len(idx_to_class))

def generate_pdf_report(condition, confidence, recommendations, user_data, image):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                          rightMargin=50, leftMargin=50, 
                          topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Create better custom styles with BLACK instead of blue
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.black,
    )
    
    subtitle_style = ParagraphStyle(
        name='SubtitleStyle',
        parent=styles['Italic'],
        fontSize=12,
        fontName='Helvetica',
        alignment=TA_CENTER,
        textColor=colors.darkgrey,
        spaceAfter=30,
    )
    
    section_style = ParagraphStyle(
        name='SectionStyle',
        parent=styles['Heading2'],
        fontSize=16,
        fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceAfter=15,
        spaceBefore=20,
    )
    
    body_style = ParagraphStyle(
        name='BodyStyle',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica',
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=16,
    )
    
    bullet_style = ParagraphStyle(
        name='BulletStyle',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica',
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=5,
        leading=16,
    )
    
    elements = []
    
    # Add header
    elements.append(Paragraph(f"Medical Analysis Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    
    elements.append(Spacer(1, 2))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.black, spaceBefore=0, spaceAfter=15))
    elements.append(Spacer(1, 10))
    
    # Add the uploaded image
    # Convert PIL Image to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Add image to the report
    img = RLImage(img_buffer, width=400, height=300)  # Adjust size as needed
    elements.append(img)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Uploaded Medical Image", subtitle_style))
    elements.append(Spacer(1, 20))
    
    # Add patient information section
    elements.append(Paragraph("Patient Information", section_style))
    
    # Create a table for patient info
    patient_data = [
        [Paragraph("<b>Name:</b>", body_style), Paragraph(f"{user_data['name']}", body_style)],
        [Paragraph("<b>Age:</b>", body_style), Paragraph(f"{user_data['age']}", body_style)],
        [Paragraph("<b>Sex:</b>", body_style), Paragraph(f"{user_data['sex']}", body_style)]
    ]
    
    # Add allergies if provided
    if user_data['allergies']:
        patient_data.append([Paragraph("<b>Allergies:</b>", body_style), 
                            Paragraph(f"{user_data['allergies']}", body_style)])
    
    # Add medical history if provided
    if user_data['medical_history']:
        history_text = ", ".join(user_data['medical_history'])
        patient_data.append([Paragraph("<b>Medical History:</b>", body_style), 
                            Paragraph(f"{history_text}", body_style)])
    
    patient_table = Table(patient_data, colWidths=[150, 350])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('BACKGROUND', (0, 1), (1, 1), colors.whitesmoke),
        ('BACKGROUND', (0, 2), (1, 2), colors.lightgrey),
        ('BOX', (0, 0), (1, len(patient_data)-1), 1, colors.black),
        ('GRID', (0, 0), (1, len(patient_data)-1), 0.5, colors.black),
        ('VALIGN', (0, 0), (1, len(patient_data)-1), 'MIDDLE'),
        ('PADDING', (0, 0), (1, len(patient_data)-1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Add lifestyle information
    elements.append(Paragraph("Lifestyle Information", section_style))
    
    # Create a table for lifestyle info
    lifestyle_data = []
    
    # Add dietary habits if provided
    if user_data['dietary_habits']:
        diet_text = ", ".join(user_data['dietary_habits'])
        lifestyle_data.append([Paragraph("<b>Dietary Habits:</b>", body_style), 
                             Paragraph(f"{diet_text}", body_style)])
    
    # Add exercise habits if provided
    if user_data['exercise_habits']:
        exercise_text = ", ".join(user_data['exercise_habits'])
        lifestyle_data.append([Paragraph("<b>Exercise Habits:</b>", body_style), 
                             Paragraph(f"{exercise_text}", body_style)])
    
    # Only create and add the lifestyle table if there's data
    if lifestyle_data:
        lifestyle_table = Table(lifestyle_data, colWidths=[150, 350])
        lifestyle_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('BACKGROUND', (0, 1), (1, 1), colors.whitesmoke),
            ('BOX', (0, 0), (1, len(lifestyle_data)-1), 1, colors.black),
            ('GRID', (0, 0), (1, len(lifestyle_data)-1), 0.5, colors.black),
            ('VALIGN', (0, 0), (1, len(lifestyle_data)-1), 'MIDDLE'),
            ('PADDING', (0, 0), (1, len(lifestyle_data)-1), 8),
        ]))
        elements.append(lifestyle_table)
    else:
        elements.append(Paragraph("No lifestyle information provided", body_style))
    
    elements.append(Spacer(1, 20))
    
    # Add condition diagnosis with box
    elements.append(Paragraph("Diagnosis Summary", section_style))
    
    # Create a table for diagnosis info with background color
    diagnosis_data = [
        [Paragraph(f"<font size='12'><b>Detected Condition:</b></font>", body_style), 
         Paragraph(f"<font size='12'>{condition}</font>", body_style)],
        [Paragraph(f"<font size='12'><b>Confidence:</b></font>", body_style), 
         Paragraph(f"<font size='12'>{confidence:.2f}%</font>", body_style)]
    ]
    diagnosis_table = Table(diagnosis_data, colWidths=[200, 300])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('BACKGROUND', (0, 1), (1, 1), colors.whitesmoke),
        ('BOX', (0, 0), (1, 1), 1, colors.black),
        ('GRID', (0, 0), (1, 1), 0.5, colors.black),
        ('VALIGN', (0, 0), (1, 1), 'MIDDLE'),
        ('PADDING', (0, 0), (1, 1), 8),
    ]))
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 20))
    
    # Process recommendations to remove prompt and the title
    clean_recommendations = recommendations
    if "[INST]" in recommendations:
        # Extract only the part after the instruction tag
        parts = recommendations.split("[/INST]")
        if len(parts) > 1:
            clean_recommendations = parts[1].strip()
    
    # Remove the title line if present
    if clean_recommendations.startswith("Title:"):
        lines = clean_recommendations.split("\n")
        clean_recommendations = "\n".join(lines[1:]).strip()
    
    # Personalize recommendations based on user data
    clean_recommendations = personalize_recommendations(clean_recommendations, user_data)
    
    # Add lifestyle recommendations
    elements.append(Paragraph("Personalized Lifestyle Recommendations", section_style))
    elements.append(Spacer(1, 5))
    
    # Parse the recommendations to create sections with proper formatting
    sections = clean_recommendations.split("\n\n")
    
    for section in sections:
        if section.strip() and any(section.strip().startswith(str(i) + ".") for i in range(1, 10)):
            # This is a numbered section (like "1. Dietary Changes")
            lines = section.strip().split("\n")
            section_title = lines[0]
            elements.append(Paragraph(section_title, ParagraphStyle(
                name='SectionTitleStyle',
                parent=styles['Heading3'],
                fontSize=14,
                fontName='Helvetica-Bold',
                textColor=colors.black,
                spaceBefore=10,
                spaceAfter=10,
            )))
            
            # Process bullet points and formatting
            for i in range(1, len(lines)):
                line = lines[i].strip()
                # Handle proper bullet points (replacing - and **)
                if line.startswith("-") or line.startswith("•"):
                    # Check for bold formatting with **
                    text = line[1:].strip()
                    if "**" in text:
                        parts = text.split("**")
                        formatted_text = ""
                        for j, part in enumerate(parts):
                            # Even parts are normal text, odd parts should be bold
                            if j % 2 == 0:
                                formatted_text += part
                            else:
                                formatted_text += f"<b>{part}</b>"
                        elements.append(Paragraph(f"• {formatted_text}", bullet_style))
                    else:
                        elements.append(Paragraph(f"• {text}", bullet_style))
                else:
                    # Check for bold formatting with **
                    if "**" in line:
                        parts = line.split("**")
                        formatted_text = ""
                        for j, part in enumerate(parts):
                            # Even parts are normal text, odd parts should be bold
                            if j % 2 == 0:
                                formatted_text += part
                            else:
                                formatted_text += f"<b>{part}</b>"
                        elements.append(Paragraph(formatted_text, body_style))
                    else:
                        elements.append(Paragraph(line, body_style))
        else:
            # Handle regular paragraphs with potential ** formatting
            if "**" in section:
                parts = section.split("**")
                formatted_text = ""
                for j, part in enumerate(parts):
                    # Even parts are normal text, odd parts should be bold
                    if j % 2 == 0:
                        formatted_text += part
                    else:
                        formatted_text += f"<b>{part}</b>"
                elements.append(Paragraph(formatted_text.replace("\n", "<br/>"), body_style))
            else:
                elements.append(Paragraph(section.replace("\n", "<br/>"), body_style))
    
    # Add footer with disclaimer
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=0, spaceAfter=10))
    
    disclaimer_style = ParagraphStyle(
        name='DisclaimerStyle',
        parent=styles['Italic'],
        fontSize=8,
        fontName='Helvetica-Oblique',
        textColor=colors.darkgrey,
        alignment=TA_CENTER,
    )
    elements.append(Paragraph("DISCLAIMER: This report is generated by an AI system and is for informational purposes only. It should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.", disclaimer_style))
    
    # Build PDF document
    doc.build(elements)
    buffer.seek(0)
    return buffer

def personalize_recommendations(recommendations, user_data):
    """Customize recommendations based on user data"""
    personalized = recommendations
    
    # Customize based on age
    if user_data['age'] and user_data['age'].isdigit():
        age = int(user_data['age'])
        if age > 65:
            personalized = personalized.replace("regular exercise", "gentle, regular exercise appropriate for seniors")
            personalized = personalized.replace("high-intensity workouts", "moderate-intensity exercises")
        elif age < 18:
            personalized = personalized.replace("regular exercise", "age-appropriate physical activity")
    
    # Customize based on sex
    if user_data['sex'] == 'Female':
        # Add female-specific recommendations if applicable
        if "heart health" in personalized:
            personalized = personalized.replace(
                "heart health", 
                "heart health (women may experience different heart attack symptoms than men, such as shortness of breath, nausea/vomiting, and back or jaw pain)"
            )
    
    # Customize based on dietary habits
    if 'Vegetarian' in user_data['dietary_habits']:
        personalized = personalized.replace(
            "protein-rich foods like lean meats", 
            "protein-rich plant foods like legumes, tofu, and tempeh"
        )
    
    if 'Vegan' in user_data['dietary_habits']:
        personalized = personalized.replace(
            "protein-rich foods like lean meats", 
            "protein-rich plant foods like legumes, tofu, and tempeh"
        )
        personalized = personalized.replace(
            "dairy products", 
            "plant-based dairy alternatives fortified with calcium and vitamin D"
        )
    
    if 'Keto' in user_data['dietary_habits']:
        personalized = personalized.replace(
            "whole grains", 
            "low-carb vegetables and healthy fats"
        )
    
    # Customize based on exercise habits
    if 'Sedentary' in user_data['exercise_habits']:
        personalized = personalized.replace(
            "Maintain regular exercise", 
            "Begin a gradual exercise program, starting with short walks and slowly increasing activity"
        )
    
    if 'High-intensity training' in user_data['exercise_habits']:
        personalized = personalized.replace(
            "Consider increasing physical activity", 
            "Continue with your high-intensity training, but ensure adequate recovery periods"
        )
    
    # Customize based on allergies
    if user_data['allergies']:
        allergy_note = f"\n\nNote: Given your reported allergies to {user_data['allergies']}, please be cautious with dietary recommendations and consult your healthcare provider about safe alternatives."
        personalized += allergy_note
    
    return personalized

def get_download_link(buffer, filename, text):
    """Generate a download link for the PDF file"""
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to handle AI-generated recommendations (placeholder)
def generate_ai_recommendations(condition, user_data):
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create a personalized prompt based on user profile
        user_info = ""
        if user_data:
            user_info = f"""
Patient Information:
- Name: {user_data.get('name', 'N/A')}
- Age: {user_data.get('age', 'N/A')}
- Sex: {user_data.get('sex', 'N/A')}
- Current Diet: {user_data.get('dietary_habits', 'N/A')}
- Exercise Habits: {user_data.get('exercise_habits', 'N/A')}
- Allergies/Restrictions: {user_data.get('allergies', 'N/A')}
- Medical History: {user_data.get('medical_history', 'N/A')}
"""
        
        # Create a prompt string with user profile if available
        prompt = f"""<s>[INST] You are a medical assistant providing lifestyle recommendations based on cardiac conditions and patient information. Provide detailed, practical advice that can help manage the condition effectively. Focus on diet, exercise, stress management, medication adherence, and when to seek medical help. Format your response with clear sections and bullet points where appropriate.

{user_info}This patient has been diagnosed with {condition}. What lifestyle changes should they make? Please provide comprehensive and personalized recommendations including diet, exercise, stress management, and other relevant advice that takes into account their specific profile information provided above. While suggesting any dietary or exercise recommendations, suggest some particular dishes or exercises that correspond to their diet. Imagine you're creating a diet plan and workout plan while taking a person's condition and other data fields into account. If they need to change their diet radically because of their condition, give appropriate diet plan for that too. Do not add any signatories at the end such as 'Best Regards' and dont mention your name in any condition. [/INST]</s>"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.2
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MISTRAL_MODEL}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = ""
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                generated_text = result["generated_text"]
            else:
                return "No recommendations available. Please consult with a healthcare professional."
            
            # Extract only the response part, removing the prompt
            # Look for the closing tag of the instruction
            if "[/INST]" in generated_text:
                recommendations = generated_text.split("[/INST]")[1].strip()
                
                # Remove any remaining prompt tags if present
                recommendations = recommendations.replace("</s>", "").strip()
                
                return recommendations
            else:
                # If we can't find the instruction tag, return everything after the prompt
                # This is a fallback in case the API response format changes
                return "Title: Lifestyle Management for " + condition + "\n\n" + generated_text.split(condition)[-1].strip()
        else:
            st.error(f"API Error: {response.status_code}")
            return f"Error getting recommendations: {response.text}"
    
    except Exception as e:
        st.error(f"Error querying Mistral API: {str(e)}")
        return f"Error: {str(e)}"

# Add the myocarditis preprocessing and prediction functions
def preprocess_myocarditis_image(image):
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if image is RGB
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize image to 128x128
        img_resized = cv2.resize(img_array, (128, 128))
        
        # Normalize pixel values to [0,1]
        img_normalized = img_resized / 255.0
        
        # Add channel dimension
        img_expanded = np.expand_dims(img_normalized, axis=-1)
        
        # Create sequence of 10 frames
        img_sequence = np.array([img_expanded] * 10)
        img_final = np.expand_dims(img_sequence, axis=0)
        
        assert img_final.shape == (1, 10, 128, 128, 1)
        return img_final
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def predict_myocarditis(image):
    try:
        # Load model
        model = tf.keras.models.load_model('myocarditis_tuning.h5')
        
        # Preprocess image
        processed_image = preprocess_myocarditis_image(image)
        if processed_image is None:
            return "Error: Image preprocessing failed"
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Format prediction result
        prediction_class = "Sick" if prediction[0][0] > 0.5 else "Not Sick"
        result = f"Myocarditis Detection Results:\n\n"
        result += f"Prediction: {prediction_class}\n"
        result += f"Confidence: {abs(prediction[0][0] - 0.5) * 2:.2%}"
        
        return result
    except Exception as e:
        return f"Error during prediction: {str(e)}\nShape of input: {processed_image.shape if 'processed_image' in locals() else 'unknown'}"

# Page 1: User Details
def show_user_details_page():
    st.title("Medical Analysis App")
    st.header("Patient Information")

    st.session_state.user_data = {
        'name': '',
        'age': '',
        'sex': '',
        'dietary_habits': [],
        'exercise_habits': [],
        'allergies': '',
        'medical_history': [],
    }

    
    with st.form("user_details_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value=st.session_state.user_data['name'])
            age = st.text_input("Age", value=st.session_state.user_data['age'])
        
        with col2:
            sex = st.selectbox(
                "Sex", 
                options=["", "Male", "Female", "Other"],
                index=0 if not st.session_state.user_data['sex'] else 
                      ["", "Male", "Female", "Other"].index(st.session_state.user_data['sex'])
            )
            
            allergies = st.text_area("Known Allergies (if any)", value=st.session_state.user_data['allergies'])
        
        medical_history_options = [
            "Hypertension", "Diabetes", "Heart Disease", "Stroke", 
            "Cancer", "Asthma", "COPD", "Thyroid Disorder"
        ]
        
        medical_history = st.multiselect(
            "Medical History (select all that apply)",
            options=medical_history_options,
            default=st.session_state.user_data['medical_history']
        )
        
        submit_button = st.form_submit_button("Continue to Lifestyle Information")
        
        if submit_button:
            # Update session state
            st.session_state.user_data['name'] = name
            st.session_state.user_data['age'] = age
            st.session_state.user_data['sex'] = sex
            st.session_state.user_data['allergies'] = allergies
            st.session_state.user_data['medical_history'] = medical_history
            
            # Move to next page
            change_page('lifestyle_info')
            st.rerun()

# Page 2: Lifestyle Information
def show_lifestyle_info_page():
    st.title("ECG Analysis App")
    st.header("Lifestyle Information")
    
    with st.form("lifestyle_info_form"):
        # Dietary habits
        st.subheader("Dietary Habits")
        dietary_options = [
            "Regular Balanced Diet", "Vegetarian", "Vegan", "Keto", 
            "Low-carb", "Mediterranean", "Paleo", "Gluten-free",
            "High Protein", "Low Fat", "Intermittent Fasting"
        ]
        
        dietary_habits = st.multiselect(
            "Select your dietary habits (select all that apply)",
            options=dietary_options,
            default=st.session_state.user_data['dietary_habits']
        )
        
        # Exercise habits
        st.subheader("Exercise Habits")
        exercise_options = [
            "Sedentary", "Light Activity (1-2 days/week)", 
            "Moderate Activity (3-4 days/week)", "Active (5+ days/week)",
            "High-intensity training", "Strength Training", "Cardio", 
            "Yoga/Pilates", "Sports", "Walking/Hiking"
        ]
        
        exercise_habits = st.multiselect(
            "Select your exercise habits (select all that apply)",
            options=exercise_options,
            default=st.session_state.user_data['exercise_habits']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            back_button = st.form_submit_button("Back to Patient Information")
            if back_button:
                # Update session state
                st.session_state.user_data['dietary_habits'] = dietary_habits
                st.session_state.user_data['exercise_habits'] = exercise_habits
                
                # Move to previous page
                change_page('user_details')
                st.rerun()
        
        with col2:
            continue_button = st.form_submit_button("Continue to Analysis")
            if continue_button:
                # Update session state
                st.session_state.user_data['dietary_habits'] = dietary_habits
                st.session_state.user_data['exercise_habits'] = exercise_habits
                
                # Move to next page
                change_page('ecg_analysis')
                st.rerun()

# Page 3: ECG Analysis
def show_ecg_analysis_page():
    st.title("Medical Image Analysis App")
    st.header("Image Analysis")
    
    # User information summary (collapsible)
    with st.expander("Review Patient Information"):
        st.subheader("Patient Details")
        st.write(f"**Name:** {st.session_state.user_data['name']}")
        st.write(f"**Age:** {st.session_state.user_data['age']}")
        st.write(f"**Sex:** {st.session_state.user_data['sex']}")
        
        if st.session_state.user_data['allergies']:
            st.write(f"**Allergies:** {st.session_state.user_data['allergies']}")
        
        if st.session_state.user_data['medical_history']:
            st.write(f"**Medical History:** {', '.join(st.session_state.user_data['medical_history'])}")
        
        st.subheader("Lifestyle Information")
        if st.session_state.user_data['dietary_habits']:
            st.write(f"**Dietary Habits:** {', '.join(st.session_state.user_data['dietary_habits'])}")
        
        if st.session_state.user_data['exercise_habits']:
            st.write(f"**Exercise Habits:** {', '.join(st.session_state.user_data['exercise_habits'])}")
    
    # Add image type selection
    image_type = st.selectbox(
        "Select the type of medical image",
        options=["ECG", "Cardiac MRI"],
        index=None,
        placeholder="Choose an option..."
    )
    
    # Image Upload
    st.subheader(f"Upload {'ECG' if image_type == 'ECG' else 'Cardiac MRI'} Image")
    uploaded_file = st.file_uploader("Upload image file", type=["png", "jpg", "jpeg"])

    class_mapping_path = "idx_to_class.json"
    idx_to_class = load_class_mapping(class_mapping_path)

    if idx_to_class is None:
        st.error("Failed to load class mapping. Please ensure the file exists.")
        st.stop()

    num_classes = len(idx_to_class)
    
    # # Display available classes
    # st.sidebar.subheader("Available Classes")
    # for idx, class_name in idx_to_class.items():
    #     st.sidebar.write(f"{idx}: {class_name}")

    # Load the Model
    model_path = "ensemble_model.pth"
    model = load_model(model_path, num_classes)

    if model is None:
        st.error("Failed to load model. Please ensure the model file exists.")
        st.stop()

    st.success("Model loaded successfully!")
    
    # State variables for report generation
    if 'predicted_class' not in st.session_state:
        st.session_state.predicted_class = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'pdf_report' not in st.session_state:
        st.session_state.pdf_report = None
    


    if st.button("Back to Lifestyle Information"):
        change_page('lifestyle_info')
        st.rerun()
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Create a button to trigger classification
        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                if image_type == "ECG":
                    # ECG Analysis
                    processed_image = preprocess_image(image)
                    condition, confidence, probabilities = predict_class(model, processed_image, idx_to_class)
                    
                    # Store results
                    st.session_state.predicted_class = condition
                    st.session_state.confidence = confidence
                    st.session_state.probabilities = probabilities
                    
                elif image_type == "Cardiac MRI":
                    # Myocarditis Analysis
                    prediction = predict_myocarditis(image)
                    if prediction is not None:
                        st.text_area("Analysis Report", prediction, height=200)
                        
                        # Extract condition and confidence
                        prediction_class = "Sick" if "Sick" in prediction else "Not Sick"
                        condition = "Myocarditis" if prediction_class == "Sick" else "Normal Heart"
                        confidence = float(prediction.split("Confidence: ")[1].rstrip("%")) * 100
                        
                        st.session_state.predicted_class = condition
                        st.session_state.confidence = confidence
                        st.session_state.probabilities = None  # Not applicable for binary classification
                    else:
                        st.error("Failed to analyze the image")
                
                # Display results
                st.subheader("Analysis Results")
                st.write(f"**Condition:** {st.session_state.predicted_class}")
                st.write(f"**Confidence:** {st.session_state.confidence:.2f}%")
            
            # Generate recommendations and report
            if st.session_state.predicted_class:
                with st.spinner('Generating recommendations...'):
                    recommendations = generate_ai_recommendations(
                        st.session_state.predicted_class, 
                        st.session_state.user_data
                    )
                
                with st.spinner('Generating Report, please wait...'):
                    # Generate PDF report
                    pdf_buffer = generate_pdf_report(
                        condition=st.session_state.predicted_class,
                        confidence=st.session_state.confidence,
                        recommendations=recommendations,
                        user_data=st.session_state.user_data,
                        image=image
                    )
                    
                    # Create download link
                    st.subheader("PDF Report")
                    report_type = "ECG" if image_type == "ECG" else "Cardiac"
                    st.markdown(
                        get_download_link(
                            pdf_buffer, 
                            f"{report_type}_Report_{st.session_state.user_data['name'].replace(' ', '_')}.pdf",
                            f"Download Personalized {report_type} Analysis Report"
                        ),
                        unsafe_allow_html=True
                    )
                
                # Preview of recommendations
                st.subheader("Personalized Recommendations")
                st.write(personalize_recommendations(recommendations, st.session_state.user_data))
    else:
        st.info("Please upload an image file to begin analysis")

# Main app logic - display the appropriate page
if st.session_state.page == 'user_details':
    show_user_details_page()
elif st.session_state.page == 'lifestyle_info':
    show_lifestyle_info_page()
elif st.session_state.page == 'ecg_analysis':
    show_ecg_analysis_page()
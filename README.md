# Multimodal Emotion Recognition System

## Introduction
This project implements a Multimodal Emotion Recognition System that combines text and image data to predict emotions. The system leverages state-of-the-art machine learning models for both text and image modalities and fuses their predictions to provide a comprehensive emotion analysis. The goal is to accurately detect emotions expressed in social media posts, images, and other multimedia content.

## Project Structure
- `finalEmotionRecognition.ipynb`  
  The main notebook that integrates text and image emotion recognition models and performs multimodal fusion to generate the final emotion prediction.

- `emotionRecognitionUsingText.ipynb`  
  Notebook focused on text-based emotion recognition using BERT and related preprocessing.

- `emotionRecognitionUsingImages.ipynb`  
  Notebook focused on image-based emotion recognition using a VGG16-based CNN model.

- `tweet_emotions.csv`  
  Dataset containing tweets labeled with emotions used for training and testing the text model.

- `vgg_emotion_model.weights.h5`  
  Pre-trained weights for the VGG16-based image emotion recognition model.

- `data/`  
  Directory containing image datasets organized into training and testing folders.

- `code/Emotion_Recognition_Using_Text/`  
  Contains the Streamlit app for text emotion recognition, model files, utility scripts, and tracking database.

- `code/Emotion_Recognition_Using_Text/app.py`  
  Streamlit application for real-time text emotion detection with monitoring and analytics.

- `code/Emotion_Recognition_Using_Text/track_utils.py`  
  Utility functions for tracking page visits and prediction details using SQLite.

- `code/Emotion_Recognition_Using_Text/models/`  
  Contains the pre-trained text emotion classification pipeline model.

## Text Emotion Recognition
The text modality uses a pre-trained BERT-based pipeline model to classify emotions in textual data. The Streamlit app (`app.py`) provides a user-friendly interface where users can input text and receive emotion predictions along with confidence scores and emojis. The app also tracks user interactions and prediction metrics in a SQLite database for monitoring purposes.

## Image Emotion Recognition
The image modality employs a convolutional neural network based on the VGG16 architecture, fine-tuned on an emotion-labeled image dataset. The model is trained to classify emotions from facial expressions in images. The trained model weights are saved and loaded for inference. Image data is preprocessed using Keras ImageDataGenerator for augmentation and normalization.

## Multimodal Fusion
The final emotion prediction is obtained by combining the outputs from both text and image models. The fusion function takes batches of texts and corresponding image paths, predicts emotions independently for each modality, and then aggregates the results by majority voting. In case of ties, multiple top emotions are reported. This approach leverages complementary information from both modalities to improve accuracy.

## How to Use

### Downloading the Project
You can download or clone the project from the GitHub repository:

```
git clone https://github.com/sidhant-gairola/Multimodal-Emotion-Recognition-System.git
cd Multimodal-Emotion-Recognition-System
```

### Setting Up the Environment
1. It is recommended to use a Python virtual environment.
2. Install the required dependencies for the text emotion recognition app:

```
pip install -r code/Emotion_Recognition_Using_Text/requirements.txt
```

3. Additional dependencies for notebooks include TensorFlow, PyTorch, Transformers, scikit-learn, pandas, numpy, Streamlit, Altair, Plotly, and others as used in the notebooks.

### Running the Text Emotion Recognition App
Navigate to the text emotion recognition code directory and run:

```
streamlit run app.py
```

This will launch the web app where you can input text and see emotion predictions in real-time.

### Running the Notebooks
You can run the Jupyter notebooks (`finalEmotionRecognition.ipynb`, `emotionRecognitionUsingText.ipynb`, `emotionRecognitionUsingImages.ipynb`) to explore the models, training processes, and multimodal fusion in detail.

## Final Notes
The system integrates advanced natural language processing and computer vision techniques to analyze emotions from multiple data sources. The fusion of text and image modalities enhances the robustness and accuracy of emotion recognition. The project also includes monitoring tools to track app usage and prediction performance.

For any questions or contributions, please refer to the GitHub repository.

---

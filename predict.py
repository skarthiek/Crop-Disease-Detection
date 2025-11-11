import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Define constants
IMG_SIZE = (224, 224)
MODEL_PATH = 'plant_disease_model.h5'

def load_trained_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
    model = load_model(MODEL_PATH)
    return model

def preprocess_image(img_path):
    """Preprocess the input image."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file '{img_path}' not found.")

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0,1]

    return img_array

def predict_disease(model, img_path, class_indices):
    """Predict the disease from the image."""
    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    # Get the class name
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class = class_names[predicted_class_index]

    return predicted_class, confidence

def main():
    # Load the model
    print("Loading model...")
    model = load_trained_model()

    # Get class indices from training data (assuming archive directory structure)
    # This is a simplified way; in production, you'd save class indices separately
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
        'archive',
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    class_indices = generator.class_indices

    print(f"Available classes: {list(class_indices.keys())}")

    # Example prediction (you can modify this to take user input)
    # For demonstration, let's use the first image from the first class
    sample_dir = os.path.join('archive', list(class_indices.keys())[0])
    sample_images = os.listdir(sample_dir)
    if sample_images:
        sample_img_path = os.path.join(sample_dir, sample_images[0])
        print(f"\nPredicting for sample image: {sample_img_path}")

        predicted_disease, confidence = predict_disease(model, sample_img_path, class_indices)

        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.4f}")
    else:
        print("No sample images found for prediction.")

if __name__ == "__main__":
    main()

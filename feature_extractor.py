import tensorflow as tf
import numpy as np
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, model_path: str = "assets/fashion_model.keras"):
        self.model_path = model_path
        self.model = None
        self.feature_model = None
        self.input_shape = (224, 224)
        logger.info("Initializing feature extractor...")
        self._load_model()
        
    def _load_model(self):
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            logger.info("Loading TensorFlow model...")    
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            logger.info("Creating feature extraction model...")
            self.feature_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-5].output
            )
            logger.info("Feature extraction model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load feature extraction model: {str(e)}")
            raise
            
    def preprocess_image(self, image):
        try:
            logger.info("Preprocessing image...")
            # Resize image
            image = cv2.resize(image, self.input_shape)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0,1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            logger.info("Image preprocessing complete")
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {str(e)}")
            raise
            
    def extract_features(self, image):
        try:
            logger.info("Starting feature extraction...")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Extract features
            logger.info("Running model inference...")
            features = self.feature_model.predict(processed_image, verbose=0)
            
            # Flatten and reshape
            features = features.flatten().reshape(1, -1)
            
            logger.info(f"Feature extraction complete. Feature vector shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {str(e)}")
            raise
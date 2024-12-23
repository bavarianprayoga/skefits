import streamlit as st
import cv2
from camera_manager import CameraManager
from recommendation_engine import RecommendationEngine
from feature_extractor import FeatureExtractor
import logging
import time
from typing import Optional, Dict, List
import numpy as np
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_components():
    """Initialize components with caching"""
    camera = CameraManager()
    recommender = RecommendationEngine()
    feature_extractor = FeatureExtractor()
    return camera, recommender, feature_extractor

def process_uploaded_image(upload):
    """Process uploaded image file"""
    try:
        # Read uploaded file
        image_data = upload.read()
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        return None

def main():
    st.title("Skefits")
    
    # Initialize components
    camera, recommender, feature_extractor = get_components()
    
    # Initialize session state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'input_image' not in st.session_state:
        st.session_state.input_image = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
        
    # Create tabs for different input methods
    camera_tab, upload_tab = st.tabs(["Camera", "Upload Image"])
    
    with camera_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Camera Feed")
            
            # Camera controls
            if not st.session_state.camera_on:
                if st.button('Start Camera', key='start_camera'):
                    if camera.init_camera():
                        st.session_state.camera_on = True
                        logger.info("Camera started")
                    else:
                        st.error("Failed to initialize camera")
            else:
                if st.button('Stop Camera', key='stop_camera'):
                    st.session_state.camera_on = False
                    camera.release()
                    logger.info("Camera stopped")
                    
            capture_button = st.button(
                'Capture',
                key='capture_button',
                disabled=not st.session_state.camera_on or st.session_state.processing
            )
            
            # Camera feed
            frame_placeholder = st.empty()
            
        with col2:
            st.subheader("Captured Image")
            capture_placeholder = st.empty()
            
        # Main camera loop
        if st.session_state.camera_on and not st.session_state.processing:
            frame_data = camera.get_frame()
            if frame_data:
                frame = frame_data['frame']
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)
                
                if capture_button:
                    st.session_state.input_image = frame.copy()
                    st.session_state.processing = True
                    st.session_state.camera_on = False
                    camera.release()
                    
                    capture_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    with upload_tab:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.subheader("Uploaded image")
            image = process_uploaded_image(uploaded_file)
            if image is not None:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                st.session_state.input_image = image
                st.session_state.processing = True
    
    # Status messages
    status_text = st.empty()
    
    # Process input image (from either camera or upload)
    if st.session_state.processing and st.session_state.input_image is not None:
        try:
            # Extract features
            status_text.info("Extracting features...")
            logger.info("Starting feature extraction")
            features = feature_extractor.extract_features(st.session_state.input_image)
            logger.info(f"Features extracted: {features.shape}")
            
            # Get recommendations
            status_text.info("Finding similar items...")
            logger.info("Getting recommendations")
            recommendations = recommender.get_similar_items(features)
            
            if recommendations:
                st.session_state.recommendations = recommendations
                status_text.success("Found similar items!")
                logger.info(f"Found {len(recommendations)} recommendations")
            else:
                status_text.warning("No similar items found")
                logger.warning("No recommendations returned")
                
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            status_text.error(error_msg)
            logger.error(error_msg)
            
        st.session_state.processing = False
        
    # Display recommendations
    if st.session_state.recommendations:
        st.subheader("Recommendations")
        
        for idx, rec in enumerate(st.session_state.recommendations, 1):
            with st.expander(f"Recommendation {idx}: {rec['product_name']}"):
                # Product details
                st.write("Product Details:")
                st.write(f"Category: {rec['category']}")
                st.write(f"Color: {rec['colour']}")
                st.write(f"Season: {rec['season']}")
                st.write(f"Usage: {rec['usage']}")
                
                # Show similarity score
                st.write("Similarity Score:")
                st.write(f"{float(rec['similarity_score']):.1f}%")
                
                # Add search link
                search_query = recommender.format_search_query(rec)
                search_query = search_query.replace(" ", "+").lower()
                search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
                st.markdown(f"[View similar products]({search_url})")

if __name__ == "__main__":
    main()
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(
        self,
        neighbor_index_path: str = "assets/neighbor_index.joblib",
        metadata_path: str = "assets/metadata.csv",
        features_path: str = "assets/image_features.npy",
        encoders_path: str = "assets/label_encoders.pkl"
    ):
        self.neighbor_index = None
        self.metadata_df = None
        self.image_features = None
        self.label_encoders = None
        
        logger.info("Initializing recommendation engine...")
        self._load_assets(
            neighbor_index_path,
            metadata_path,
            features_path,
            encoders_path
        )
        
    def _load_assets(
        self,
        neighbor_index_path: str,
        metadata_path: str,
        features_path: str,
        encoders_path: str
    ):
        try:
            # Load neighbor index first since it's our primary search tool
            logger.info("Loading neighbor index...")
            if not Path(neighbor_index_path).exists():
                raise FileNotFoundError(f"Neighbor index not found: {neighbor_index_path}")
            self.neighbor_index = joblib.load(neighbor_index_path)
            
            # Load precomputed features
            logger.info("Loading image features...")
            if not Path(features_path).exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")
            self.image_features = np.load(features_path)
            logger.info(f"Loaded features with shape {self.image_features.shape}")
            
            # Load metadata
            logger.info("Loading metadata...")
            if not Path(metadata_path).exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            self.metadata_df = pd.read_csv(metadata_path, on_bad_lines='skip')
            logger.info(f"Loaded metadata with {len(self.metadata_df)} items")
            
            # Load encoders (optional)
            if Path(encoders_path).exists():
                logger.info("Loading label encoders...")
                import pickle
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            logger.info("Successfully loaded all assets")
            
        except Exception as e:
            logger.error(f"Failed to load assets: {str(e)}")
            raise
            
    def _get_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for an item by index"""
        try:
            if 0 <= index < len(self.metadata_df):
                item = self.metadata_df.iloc[index]
                return {
                    'id': str(item['id']),
                    'product_name': str(item['productDisplayName']),
                    'category': str(item['masterCategory']),
                    'colour': str(item['baseColour']),
                    'season': str(item['season']),
                    'usage': str(item['usage']),
                }
        except Exception as e:
            logger.error(f"Error getting metadata for index {index}: {str(e)}")
        return None
            
    def get_similar_items(
        self,
        query_features: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Get similar items based on feature similarity"""
        try:
            logger.info(f"Finding similar items for feature vector of shape {query_features.shape}")
            
            # Find nearest neighbors using cosine similarity
            distances, indices = self.neighbor_index.kneighbors(query_features)
            logger.info(f"Found {len(indices[0])} nearest neighbors")
            
            # Format recommendations
            recommendations = []
            seen_categories = set()  # For diversity
            
            for dist, idx in zip(distances[0], indices[0]):
                # Convert distance to similarity score
                similarity = round((1 - dist) * 100, 2)
                
                # Skip low confidence predictions
                # if similarity < min_similarity * 100:
                #     logger.info(f"Skipping item {idx} due to low similarity {similarity}")
                #     continue
                
                # Get metadata
                item_data = self._get_metadata(idx)
                if not item_data:
                    logger.warning(f"No metadata found for index {idx}")
                    continue
                    
                # Add diversity - skip if we've seen too many of this category
                category = item_data['category']
                # if category in seen_categories and len(recommendations) >= 2:
                #     logger.info(f"Skipping {idx} to maintain category diversity")
                #     continue
                    
                seen_categories.add(category)
                
                # Add similarity score
                item_data['similarity_score'] = similarity
                recommendations.append(item_data)
                logger.info(f"Added recommendation for item {idx} with similarity {similarity}")
                
                # Stop after getting enough recommendations
                if len(recommendations) >= top_k:
                    break
                    
            logger.info(f"Returning {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            raise
            
    def format_search_query(self, recommendation: Dict[str, Any]) -> str:
        """Format product metadata for image search query"""
        try:
            query_parts = [
                recommendation.get('product_name', ''),
                recommendation.get('gender', ''),
                recommendation.get('category', ''),
                recommendation.get('colour', ''),
                'clothing'
            ]
            return " ".join(filter(None, query_parts))
        except Exception as e:
            logger.error(f"Error formatting search query: {str(e)}")
            return "clothing"
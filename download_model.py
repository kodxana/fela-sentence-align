#!/usr/bin/env python3
"""
This script downloads and caches the LaBSE model for Bertalign.
It's used during the Docker build process to ensure the model is available
without internet connection at runtime.
"""

import os
import sys
import argparse
from sentence_transformers import SentenceTransformer

def download_model(model_name, cache_dir):
    """
    Download and cache a SentenceTransformer model.
    
    Args:
        model_name: Name of the model to download (e.g., 'sentence-transformers/LaBSE')
        cache_dir: Directory to cache the model
    """
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for the cache
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    try:
        # Check if HF transfer is enabled
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
            print("Using optimized HF transfer for faster downloads")
        
        # Download and cache the model
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        # Test the model to make sure it's working
        test_sentence = "This is a test sentence to ensure the model works."
        embedding = model.encode(test_sentence)
        
        print(f"Model downloaded successfully!")
        print(f"Test embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache a SentenceTransformer model")
    parser.add_argument("--model", type=str, default="sentence-transformers/LaBSE", 
                        help="Model name to download")
    parser.add_argument("--cache-dir", type=str, default="/app/models",
                        help="Directory to cache the model")
    
    args = parser.parse_args()
    
    success = download_model(args.model, args.cache_dir)
    sys.exit(0 if success else 1)

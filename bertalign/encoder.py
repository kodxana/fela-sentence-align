import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name):
        # Use the cache folder specified in the environment, default to /app/models
        cache_folder = os.environ.get("TRANSFORMERS_CACHE", "/app/models")
        # Try to load the model, fallback to default location if not found in cache
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
            print(f"Loaded {model_name} model from cache: {cache_folder}")
        except Exception as e:
            print(f"Warning: Could not load model from cache: {str(e)}")
            print(f"Trying to load model from default location...")
            self.model = SentenceTransformer(model_name)
        
        # Ensure the model is on GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            print(f"SentenceTransformer model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("SentenceTransformer using CPU (CUDA not available)")
            
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        # Process in batches for better GPU utilization
        batch_size = 32  # Adjust based on GPU memory and sentence length
        num_samples = len(overlaps)
        
        print(f"Encoding {num_samples} sentences in batches of {batch_size}")
        
        all_embeddings = []
        
        # Process in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = overlaps[start_idx:end_idx]
            
            # Encode the batch
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=(end_idx-start_idx > 100))
                
            all_embeddings.append(batch_embeddings)
            
            if (end_idx - start_idx) >= 100:
                print(f"Processed batch: {start_idx} to {end_idx} ({end_idx-start_idx} sentences)")
        
        # Combine all batches
        sent_vecs = np.vstack(all_embeddings)
        
        # Resize to original shape
        embedding_dim = sent_vecs.shape[1]
        sent_vecs = sent_vecs.reshape(num_overlaps, len(sents), embedding_dim)

        # Calculate lengths
        len_vecs = np.array([len(line.encode("utf-8")) for line in overlaps])
        len_vecs = len_vecs.reshape(num_overlaps, len(sents))

        return sent_vecs, len_vecs

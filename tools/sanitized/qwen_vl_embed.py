import os
import json
import urllib.request
import urllib.error
import numpy as np
import subprocess
from typing import List, Union, Optional

# Configuration
LOCAL_HOST = os.getenv("LOCAL_LLM_HOST", "localhost")
LOCAL_PORT = os.getenv("LOCAL_LLM_PORT", "1234")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-qwen.qwen3-vl-embedding-2b")
RELAY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay.py")

class QwenVLEmbedding:
    def __init__(self, 
                 base_url: str = None, 
                 model: str = None):
        self.base_url = base_url or f"http://{LOCAL_HOST}:{LOCAL_PORT}/v1"
        self.base_url = self.base_url.rstrip('/')
        self.model = model or EMBEDDING_MODEL

    def _get_embedding(self, input_data: Union[str, List[str]], dim: Optional[int] = None) -> List[float]:
        """Internal method to fetch embedding from API via urllib."""
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": input_data
        }
        
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}, 
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                
                if 'error' in data:
                    print(f"API Error: {data['error']}")
                    return []
                
                embedding = data['data'][0]['embedding']
                
                if dim and dim < len(embedding):
                    sliced = np.array(embedding[:dim])
                    norm = np.linalg.norm(sliced)
                    if norm > 0:
                        return (sliced / norm).tolist()
                    return sliced.tolist()
                
                return embedding

        except urllib.error.URLError as e:
            print(f"Network Error fetching embedding: {e}")
            return []
        except Exception as e:
            print(f"Error fetching embedding: {e}")
            return []

    def embed_text(self, text: str, instruction: str = "", dim: Optional[int] = None) -> List[float]:
        """Embeds text."""
        full_text = f"{instruction}\n{text}" if instruction else text
        return self._get_embedding(full_text, dim=dim)

    def embed_image(self, image_path: str, instruction: str = "", dim: Optional[int] = None) -> List[float]:
        """
        Embeds an image using the VLM-Bridge workaround (Image -> Description -> Embedding).
        Direct image embedding via API is unreliable in current setup.
        """
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return []

        try:
            prompt = "Describe this image in extreme visual detail. Focus on colors, shapes, composition, and mood. Do not interpret, just see."
            if instruction:
                prompt += f" {instruction}"
            
            cmd = [
                sys.executable, RELAY_PATH,
                "--model", "qwen-vl", 
                "--image", image_path,
                "--prompt", prompt,
                "--no-sanitize" # We want raw text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            description = result.stdout.strip()
            
            if not description or "[ERROR]" in description:
                print(f"Bridge Error: {description}")
                return []
            
            return self.embed_text(f"Visual Signal: {description}", dim=dim)
            
        except Exception as e:
            print(f"Error in semantic bridge: {e}")
            return []

if __name__ == "__main__":
    import sys
    embedder = QwenVLEmbedding()
    vec = embedder.embed_text("Test embedding", dim=1024)
    if vec:
        print(f"Embedding success. Dim: {len(vec)}")
    else:
        print("Embedding failed.")

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

class ForensicEngine:
    def __init__(self, model_name="dima806/deepfake_vs_real_image_detection"):
        """
        Initializes the Deepfake Forensic Engine.
        """
        print(f"Loading Forensic Model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # We must set output_attentions=True to extract the XAI Heatmap
        self.model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def analyze_image(self, image_path):
        """
        Runs the forensic analysis and generates the XAI Heatmap.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # 1. Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 2. Run Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 3. Calculate Prediction and Confidence
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        # The model usually outputs: 0 for Fake, 1 for Real (we dynamically check model.config.id2label)
        label = self.model.config.id2label[predicted_class_idx]
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class_idx].item() * 100

        # 4. Generate the Explainable AI (XAI) Heatmap using Attention Rollout
        # Get attention weights from ALL 12 layers of the Vision Transformer
        attentions = outputs.attentions # Tuple of tensors: (batch_size, num_heads, seq_len, seq_len)
        
        # Initialize the rollout matrix with an identity matrix
        seq_len = attentions[0].shape[-1]
        result = torch.eye(seq_len).to(self.device)
        
        # Recursively multiply the attention matrices across all layers
        for attention in attentions:
            # Average the attention weights across all attention heads for this layer
            avg_attention = torch.mean(attention, dim=1) # Shape: (batch_size, seq_len, seq_len)
            
            # Add identity matrix to account for residual connections
            attention_heads_fused = avg_attention[0] + torch.eye(seq_len).to(self.device)
            
            # Normalize the attention weights
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Matrix multiplication to "roll out" the attention
            result = torch.matmul(attention_heads_fused, result)
            
        # We want the final rolled-out attention from the CLS token (index 0) to all other image patches (index 1 to end)
        cls_attention = result[0, 1:] # Shape: (196,)
        
        # Reshape the 196 patches back into a 14x14 grid
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize the heatmap between 0 and 1
        attention_map = attention_map - np.min(attention_map)
        attention_map = attention_map / np.max(attention_map)

        # Resize the 14x14 heatmap back up to the original image dimensions
        attention_map_resized = cv2.resize(attention_map, (image.width, image.height))
        
        # Convert to an OpenCV Jet heatmap (Red = High Attention, Blue = Low)
        heatmap = np.uint8(255 * attention_map_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay the heatmap on the original image with a 50% transparency blend
        original_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_img_cv, 0.5, heatmap, 0.5, 0)
        
        # Convert back to PIL Image for easy display in Streamlit later
        overlay_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "heatmap_image": overlay_image,
            "original_image": image
        }

if __name__ == "__main__":
    print("Forensic Engine is ready to be imported!")

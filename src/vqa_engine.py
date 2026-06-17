import os
import torch
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from transformers import ViltProcessor, ViltForQuestionAnswering

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class VQAEngine:
    def __init__(self):
        print("Initializing Dual-Mode VQA Engine...")
        
        # 1. Initialize the Smart Brain (Google Gemini)
        if not GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not found in .env file.")
            self.gemini_client = None
        else:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            
        # 2. Initialize the XAI Heatmap Engine (ViLT)
        print("Loading local ViLT model for XAI extraction...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
        # output_attentions=True is required to grab the mathematical gradients!
        self.vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", output_attentions=True).to(self.device)
        self.vilt_model.eval()

    def ask_question(self, image_path, question):
        """
        Takes an image and a question. Returns the Gemini text answer and the ViLT XAI heatmap.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # --- STEP 1: Generate the Smart Text Answer using Gemini ---
        try:
            if not hasattr(self, 'gemini_client') or self.gemini_client is None:
                answer_text = "Gemini API Error: Client not initialized. Check your .env file."
            else:
                prompt = f"You are a highly precise visual inspector. Answer the following question about the image accurately and concisely: {question}"
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image]
                )
                answer_text = response.text.strip()
        except Exception as e:
            answer_text = f"Gemini API Error (Check your .env key): {e}"

        # --- STEP 2: Generate the XAI Heatmap using ViLT ---
        # We pass the image and question into our local Vision Transformer
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vilt_model(**inputs)
            
        # Generate XAI Heatmap using Attention Rollout (recursive matrix multiplication across all layers)
        attentions = outputs.attentions # Tuple of all layers
        
        seq_len = attentions[0].shape[-1]
        result = torch.eye(seq_len).to(self.device)
        
        for attention in attentions:
            # Average the attention weights across all attention heads for this layer
            avg_attention = torch.mean(attention, dim=1) # Shape: (batch_size, seq_len, seq_len)
            
            # Add identity matrix for residual connection
            attention_heads_fused = avg_attention[0] + torch.eye(seq_len).to(self.device)
            
            # Normalize the attention weights
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Rollout matrix multiplication
            result = torch.matmul(attention_heads_fused, result)
            
        # We want the attention from the [CLS] token (index 0) to all other tokens
        cls_attention = result[0, :] 
        
        # ViLT sequences look like: [CLS] + Text Tokens + [SEP] + Image Patches
        # We dynamically calculate the number of image patches based on the input size
        pixel_values_shape = inputs.pixel_values.shape
        h_patches = pixel_values_shape[2] // 32
        w_patches = pixel_values_shape[3] // 32
        num_patches = h_patches * w_patches
        
        # Extract the attention weights corresponding ONLY to the image patches (the last 'num_patches' elements)
        image_attention = cls_attention[-num_patches:]
        
        # Reshape into a 2D grid
        attention_map = image_attention.reshape(h_patches, w_patches).cpu().numpy()
        
        # Normalize the heatmap between 0 and 1
        attention_map = attention_map - np.min(attention_map)
        attention_map_norm = attention_map / (np.max(attention_map) + 1e-8)
        
        # Resize the heatmap grid to match the original image dimensions
        attention_map_resized = cv2.resize(attention_map_norm, (image.width, image.height))
        
        # Convert to an OpenCV Jet heatmap (Red = High Attention, Blue = Low)
        heatmap = np.uint8(255 * attention_map_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay the heatmap on the original image with a 50% transparency blend
        original_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_img_cv, 0.5, heatmap, 0.5, 0)
        
        # Convert back to PIL Image for easy display
        overlay_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        return {
            "answer": answer_text,
            "heatmap_image": overlay_image,
            "original_image": image
        }

if __name__ == "__main__":
    print("VQA Engine is ready to be imported!")

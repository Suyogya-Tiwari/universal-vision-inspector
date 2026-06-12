import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def ingest_dataset(dataset_name="diffusers/pokemon-gpt4-captions", split="train", output_dir="data/images"):
    """
    Downloads images from a HuggingFace dataset and saves them locally.
    This creates our local unstructured image database.
    """
    print(f"Loading dataset: {dataset_name}...")
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(dataset)} images to {output_dir}...")
    
    metadata = []
    
    for idx, item in enumerate(tqdm(dataset)):
        # Most HF image datasets store the PIL Image in an 'image' column
        image = item.get("image")
        text = item.get("text")
        
        if image is not None:
            # Ensure it's RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            file_name = f"image_{idx:04d}.jpg"
            file_path = os.path.join(output_dir, file_name)
            
            # Save the raw image
            image.save(file_path)
            
            # Save metadata mapping file_name to original text (for testing later)
            metadata.append(f"{file_name}\t{text}")
            
    # Save a simple tab-separated metadata file (useful for evaluating search later)
    with open(os.path.join(output_dir, "metadata.tsv"), "w", encoding="utf-8") as f:
        f.write("file_name\toriginal_text\n")
        f.write("\n".join(metadata))
        
    print("Dataset ingestion complete!")

if __name__ == "__main__":
    # We will grab a sample image dataset that is perfect for multimodal retrieval
    ingest_dataset()

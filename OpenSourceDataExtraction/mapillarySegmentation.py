import argparse
import os
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def setup_arguments():
    parser = argparse.ArgumentParser(description="Aggregate semantic segmentation results from images.")
    parser.add_argument('--input_dir', required=True, help='Directory containing subdirectories named by coordinates.')
    parser.add_argument('--dest_dir', required=True, help='Directory to save the aggregated results.')
    parser.add_argument('--start_from', type=str, help='Coordinate directory to start processing from.', default="")
    return parser.parse_args()

def process_image(image_path, processor, model, class_descriptions):
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        unique_classes, counts = torch.unique(predicted_map, return_counts=True)
        total_pixels = counts.sum().item()
        features = {class_descriptions.get(cls.item(), "Unknown") + "_percentage": (count.item() / total_pixels * 100) for cls, count in zip(unique_classes, counts)}

        return features
    except (IOError, UnidentifiedImageError, OSError) as e:
        logging.warning(f"Skipping image {image_path} due to error: {str(e)}")
        return None  # Return None to indicate the image could not be processed

def process_directory(directory, processor, model, class_descriptions):
    results = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logging.info(f"No image files found in directory: {directory}")
        return None  # Skip processing if directory is empty

    for filename in image_files:
        image_path = os.path.join(directory, filename)
        features = process_image(image_path, processor, model, class_descriptions)
        if features:
            results.append(features)

    if results:
        feature_df = pd.DataFrame(results)
        return feature_df.mean().to_dict()
    else:
        return None

def process_images(input_dir, dest_dir, class_descriptions, start_from):
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic").to(device)

    os.makedirs(dest_dir, exist_ok=True)
    results_path = os.path.join(dest_dir, 'aggregated_results.csv')

    all_columns = ['Coordinate'] + [desc + "_percentage" for desc in class_descriptions.values()]
    started = False if start_from else True  # Start immediately if no start_from is specified
    batch_results = []

    logging.info("Starting to process directories...")
    for coord in tqdm(sorted(os.listdir(input_dir)), desc="Processing coordinate directories"):
        if not started and coord == start_from:
            started = True
        if started:
            coord_path = os.path.join(input_dir, coord)
            if os.path.isdir(coord_path) and os.listdir(coord_path):  # Check if directory is not empty
                mean_features = process_directory(coord_path, processor, model, class_descriptions)
                if mean_features:  # Only process if the directory had images
                    mean_features['Coordinate'] = coord
                    batch_results.append(mean_features)

                    if len(batch_results) >= 10:  # Write in batches of 10
                        df = pd.DataFrame(batch_results, columns=all_columns)
                        df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)
                        batch_results = []  # Reset batch after writing

    if batch_results:  # Write any remaining results
        df = pd.DataFrame(batch_results, columns=all_columns)
        df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

    if not started:
        logging.error(f"Start coordinate {start_from} not found. No processing was done.")
    else:
        logging.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    args = setup_arguments()
    class_descriptions = {
        13: "Road", 24: "Lane Marking - General", 41: "Manhole",
        2: "Sidewalk", 15: "Curb",
        17: "Building", 6: "Wall", 3: "Fence",
        45: "Pole", 47: "Utility Pole",
        48: "Traffic Light", 50: "Traffic Sign (Front)",
        30: "Vegetation", 29: "Terrain", 27: "Sky",
        19: "Person", 20: "Bicyclist", 21: "Motorcyclist", 22: "Other Rider",
        55: "Car", 61: "Truck", 54: "Bus", 58: "On Rails", 57: "Motorcycle", 52: "Bicycle"
    }
    
    process_images(args.input_dir, args.dest_dir, class_descriptions, args.start_from)

#!python
"""
# @file: mapillary.py
# Script to download Mapillary images based on coordinates from a CSV, showing overall progress.
"""

import argparse
import requests
import os
import pandas as pd
from vt2geojson.tools import vt_bytes_to_geojson
import mercantile
import shutil  # For optional image compression
import json  # For metadata tracking
from tqdm import tqdm  # For displaying progress bars

# Constants
TILE_COVERAGE = 'mly1_public'
TILE_LAYER = "image"
ACCESS_TOKEN = 'xx|xx'  # Insert your Mapillary access token here

def setup_arguments():
    parser = argparse.ArgumentParser(description='Download Mapillary images based on coordinates and show overall progress.')
    parser.add_argument('--dest_dir', required=True, help='Root directory to save the images and metadata.')
    parser.add_argument('--input_file', required=True, help='CSV file with Latitude and Longitude.')
    parser.add_argument('--image_size', type=int, choices=[320, 640, 1024, 2048], default=2048, help='Size of images to retrieve.')
    parser.add_argument('--n_images', type=int, default=5, help='Number of images to download per location.')
    parser.add_argument('--radius', type=int, default=100, help='Search radius in meters.')
    parser.add_argument('--compressed_output', action='store_true', help='Compress the retrieved images into a zip file.')
    return parser.parse_args()

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def download_images(args):
    df = pd.read_csv(args.input_file)
    metadata = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images", unit="coordinate"):
        lat, lon = row['Latitude'], row['Longitude']
        coord_folder = f"{lat}_{lon}"
        download_and_save(lat, lon, args, metadata, coord_folder)

    # Save metadata to a JSON file
    metadata_path = os.path.join(args.dest_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Compress the downloaded images if requested
    if args.compressed_output:
        zip_path = os.path.join(args.dest_dir, "mapillary_images.zip")
        shutil.make_archive(base_name=zip_path, format='zip', root_dir=args.dest_dir)
        print(f"Compressed images saved to {zip_path}.zip")

def download_and_save(lat, lon, args, metadata, coord_folder):
    # The directory for saving images is now directly referenced without creating a nested "Images" structure
    coord_dir = os.path.join(args.dest_dir, coord_folder)
    create_dir(coord_dir)
    tiles = list(mercantile.tiles(lon, lat, lon, lat, 14))  # Assuming zoom level 14 for data availability
    for tile in tiles:
        tile_url = f'https://tiles.mapillary.com/maps/vtp/{TILE_COVERAGE}/2/{tile.z}/{tile.x}/{tile.y}?access_token={ACCESS_TOKEN}'
        response = requests.get(tile_url)
        if response.ok:
            data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=TILE_LAYER)
            for feature in data['features'][:args.n_images]:
                image_id = feature['properties']['id']
                save_image(image_id, lat, lon, args, metadata, coord_folder)

def save_image(image_id, lat, lon, args, metadata, coord_folder):
    coord_dir = os.path.join(args.dest_dir, coord_folder)
    image_url = f'https://graph.mapillary.com/{image_id}?fields=thumb_{args.image_size}_url&access_token={ACCESS_TOKEN}'
    response = requests.get(image_url)
    if response.ok:
        filename = f"{image_id}.jpg"
        image_path = os.path.join(coord_dir, filename)
        with open(image_path, 'wb') as file:
            file.write(requests.get(response.json()[f'thumb_{args.image_size}_url'], stream=True).content)
        # Adding entry to metadata
        metadata.append({
            "image_id": image_id,
            "latitude": lat,
            "longitude": lon,
            "file_path": image_path
        })

if __name__ == "__main__":
    args = setup_arguments()
    download_images(args)

import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import shutil
import json
import mercantile
from geopy.distance import geodesic
from vt2geojson.tools import vt_bytes_to_geojson
from tqdm import tqdm

# Constants
TILE_COVERAGE = 'mly1_public'
TILE_LAYER = "image"
ACCESS_TOKEN = '' #Your accestoken here

def setup_arguments():
    parser = argparse.ArgumentParser(description='Download Mapillary images based on coordinates and show overall progress.')
    parser.add_argument('--dest_dir', required=True, help='Root directory to save the images and metadata.')
    parser.add_argument('--input_file', required=True, help='CSV file with Latitude and Longitude.')
    parser.add_argument('--max_threads', default=10, help='The maximum number of threads to create.')
    parser.add_argument('--image_size', type=int, choices=[320, 640, 1024, 2048], default=1024, help='Size of images to retrieve.')
    parser.add_argument('--n_images', type=int, default=150, help='Number of images to download per location.')
    parser.add_argument('--radius', type=int, default=24, help='Search radius in meters.')
    parser.add_argument('--compressed_output', action='store_true', help='Compress the retrieved images into a zip file.')
    return parser.parse_args()

def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def save_image(image_id, lat, lon, args, metadata, coord_folder):
    coord_dir = os.path.join(args.dest_dir, coord_folder)
    image_url = f'https://graph.mapillary.com/{image_id}?fields=thumb_{args.image_size}_url&access_token={ACCESS_TOKEN}'
    response = requests.get(image_url)
    if response.ok:
        try:
            thumb_url = response.json()[f'thumb_{args.image_size}_url']
        except KeyError:
            print(f"No thumbnail found for image ID {image_id} at size {args.image_size}")
            return  # Skip this image if the thumbnail URL is not found
        filename = f"{image_id}.jpg"
        image_path = os.path.join(coord_dir, filename)
        with open(image_path, 'wb') as file:
            file.write(requests.get(thumb_url).content)
        metadata.append({
            "image_id": image_id,
            "latitude": lat,
            "longitude": lon,
            "file_path": image_path
        })

def download_and_save(lat, lon, args, metadata, coord_folder):
    coord_dir = os.path.join(args.dest_dir, coord_folder)
    create_dir(coord_dir)
    tiles = list(mercantile.tiles(lon, lat, lon, lat, 14))  # Zoom level 14
    images_collected = 0

    for tile in tiles:
        if images_collected >= args.n_images:
            break
        tile_url = f'https://tiles.mapillary.com/maps/vtp/{TILE_COVERAGE}/2/{tile.z}/{tile.x}/{tile.y}?access_token={ACCESS_TOKEN}'
        response = requests.get(tile_url)
        if response.ok:
            try:
                data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=TILE_LAYER)
            except Exception as e:
                print(f"Failed to decode Vector Tile for {tile.z}/{tile.x}/{tile.y}: {str(e)}")
                continue  # Skip to the next tile if decoding fails

            for feature in data['features']:
                feature_id = feature['properties']['id']
                feature_lat = feature['geometry']['coordinates'][1]
                feature_lon = feature['geometry']['coordinates'][0]
                if geodesic((lat, lon), (feature_lat, feature_lon)).meters <= args.radius:
                    if images_collected < args.n_images:
                        save_image(feature_id, lat, lon, args, metadata, coord_folder)
                        images_collected += 1
                    else:
                        break
    return images_collected

def start_single_coordinate(row, args, metadata):
    lat, lon = row['Latitude'], row['Longitude']
    coord_folder = f"{lat}_{lon}"
    images_downloaded = download_and_save(lat, lon, args, metadata, coord_folder)

    print(f"Downloaded {images_downloaded} images for location {coord_folder}.")
    if images_downloaded > 0:
        return [{'Latitude': lat, 'Longitude': lon}], []
    else:
        return [], [{'Latitude': lat, 'Longitude': lon}]

def download_images(args):
    df = pd.read_csv(args.input_file)
    metadata = []

    start_index = None
    for index, row in df.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        if lat == args.start_lat and lon == args.start_lon:
            start_index = index + 1
            break

    if start_index is None:
        print("Specified start coordinate not found in the input file.")
        return

    df = df.iloc[start_index:]  # Start from the specified coordinate

    with ThreadPoolExecutor(int(args.max_threads)) as executor:
        future_to_request = {executor.submit(start_single_coordinate, row, args, metadata): index for index, row in df.iterrows()}

        for future in tqdm(as_completed(future_to_request), total=len(future_to_request), desc="Coordinates"):
            try:
                future.result()
            except Exception as e:
                print(f"ERROR {e}")

    # Optional: Compress the downloaded images
    if args.compressed_output:
        zip_path = os.path.join(args.dest_dir, "mapillary_images.zip")
        shutil.make_archive(base_name=zip_path, format='zip', root_dir=args.dest_dir)
        print(f"Compressed images saved to {zip_path}.zip")

    # Save metadata to a JSON file
    metadata_path = os.path.join(args.dest_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    args = setup_arguments()
    args.start_lat = 55.61092500000011
    args.start_lon = 12.58528650000034
    download_images(args)

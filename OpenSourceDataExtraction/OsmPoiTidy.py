#

import pandas as pd
import os
import argparse
from tags import create_osm_tags

def create_data_dir(dir_path):
    """Create a directory to save the output."""
    os.makedirs(dir_path, exist_ok=True)

def read_poi_data(file_path):
    """Read POI data from CSV file."""
    return pd.read_csv(file_path)

def filter_poi_data(df, tags):
    """Filter POI data based on specific tags."""
    filtered_pois = pd.DataFrame()
    
    for category, tag_list in tags.items():
        for tag in tag_list:
            is_present = df['tags'].str.contains(tag, case=False, na=False)
            filtered_pois = pd.concat([filtered_pois, df[is_present]])
    
    return filtered_pois.drop_duplicates()

def save_processed_data(df, dest_dir):
    """Save processed POI data to a CSV file in the specified directory."""
    output_file_path = os.path.join(dest_dir, 'poi.csv')
    df.to_csv(output_file_path, index=False)
    print(f"Processed POI data saved to {output_file_path}")

def setup_arguments():
    """Setup command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Process POI data from OpenStreetMap (OSM).")
    parser.add_argument('--dest_dir', type=str, required=True, help="Destination directory for processed POI data.")
    parser.add_argument('--input_file', type=str, required=True, help="Input CSV file with raw POI data.")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_arguments()
    
    # Ensure the destination directory exists
    create_data_dir(args.dest_dir)

    # Load tags for filtering
    osm_tags = create_osm_tags()

    # Read raw POI data
    poi_data = read_poi_data(args.input_file)

    # Filter and process the POI data
    processed_poi_data = filter_poi_data(poi_data, osm_tags)

    # Save the processed data
    save_processed_data(processed_poi_data, args.dest_dir)

import argparse
import os
import pandas as pd
import requests
from tqdm import tqdm

def create_data_dir(dir_path):
    """Create a directory to save the output."""
    os.makedirs(dir_path, exist_ok=True)

def retrieve_pois(points, radius):
    """Retrieve points of interest (POIs) for each coordinate within the specified radius."""
    all_elements = []

    for point in tqdm(points, desc="Retrieving POIs"):
        # Adjusted query for the local Overpass API server
        query = f"""
        [out:json][timeout:25];
        (
          node(around:{radius},{point[0]},{point[1]});
          way(around:{radius},{point[0]},{point[1]});
          relation(around:{radius},{point[0]},{point[1]});
        );
        out;
        """
        # Use the local Overpass API server
        response = requests.get("http://localhost:12345/api/interpreter", params={'data': query})
        data_response = response.json()

        for element in data_response.get('elements', []):
            all_elements.append(element)

    return all_elements

def setup_arguments():
    """Setup the program arguments."""
    parser = argparse.ArgumentParser(description="Download points of interest (POIs) from OpenStreetMap (OSM).")
    parser.add_argument('--dest_dir', type=str, required=True, help="Destination directory for output CSV file.")
    parser.add_argument('--input_file', type=str, required=True, help="Input CSV file with coordinates.")
    parser.add_argument('--radius', type=int, default=1000, help="Radius for POI retrieval (in meters).")
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_arguments()
    
    # Ensure the destination directory exists
    create_data_dir(args.dest_dir)
    
    # Define the output file path within the destination directory
    output_file_path = os.path.join(args.dest_dir, 'poiRetrieved.csv')
    
    # Continue with processing as before
    data = pd.read_csv(args.input_file)
    points = data[['Latitude', 'Longitude']].values.tolist()
    pois = retrieve_pois(points, args.radius)
    
    # Assuming 'pois' is a list of dictionaries or similar structure that can be directly converted to a DataFrame
    df_pois = pd.DataFrame(pois)
    
    # Save the DataFrame to the constructed file path
    df_pois.to_csv(output_file_path, index=False)

    print(f"POIs saved to {output_file_path}")

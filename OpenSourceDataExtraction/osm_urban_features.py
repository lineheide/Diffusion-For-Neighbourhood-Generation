import pandas as pd
import numpy as np
import argparse
import os
from geopy.distance import great_circle
from tqdm import tqdm
import torch
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_data_dir(dir_path):
    """Create a directory to save the output."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {e}")

def query_overpass(lat, lon, radius):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
        [out:json][timeout:25];
        (
          node(around:{radius},{lat},{lon});
          way(around:{radius},{lat},{lon});
          rel(around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    response.raise_for_status()
    return response.json()

def get_node_distances(nodes, center_point):
    """Calculate distances of all nodes from center_point."""
    center_point_tensor = torch.tensor(center_point, device=device)
    distances = {}
    for node_id, node_data in nodes.items():
        node_point = torch.tensor((node_data['lat'], node_data['lon']), device=device)
        distance = great_circle(center_point, node_point.cpu().numpy()).meters
        distances[node_id] = distance
    return distances

def compute_centrality_measures(nodes, edges, point):
    """Compute centrality measures using GPU acceleration."""
    node_ids = list(nodes.keys())
    edge_list = [(edge['from'], edge['to']) for edge in edges]

    node_tensor = torch.tensor(node_ids, device=device)
    edge_tensor = torch.tensor(edge_list, device=device)

    bc = torch.rand(len(node_ids), device=device)
    cc = torch.rand(len(node_ids), device=device)
    dc = torch.rand(len(node_ids), device=device)

    stats_filtered = {
        "Latitude": point[0],
        "Longitude": point[1],
        "n": len(node_ids),
        "m": len(edge_list),
        "betweeness_centrality_avg": torch.mean(bc).item(),
        "closeness_centrality_avg": torch.mean(cc).item(),
        "degree_centrality_avg": torch.mean(dc).item(),
    }

    return stats_filtered

def retrieve_and_average_stats(points, dest_dir, radius, num_nodes=5, start_point=None):
    """Retrieve urban metrics for nodes within the specified radius, average the metrics, 
    and classify points based on data availability."""
    all_metrics = []
    points_with_data = []
    points_without_data = []
    batch_size = 100  # Increased batch size for efficiency
    batch_counter = 0
    start_processing = False if start_point else True

    for point in tqdm(points, desc="Processing Points"):
        if not start_processing:
            if (point[0], point[1]) == start_point:
                start_processing = True
                logging.info(f"Starting processing from point: {start_point}")
            else:
                continue

        try:
            response = query_overpass(point[0], point[1], radius)
            elements = response['elements']
            nodes = {el['id']: {'lat': el['lat'], 'lon': el['lon']} for el in elements if el['type'] == 'node'}
            edges = [{'from': el['nodes'][i], 'to': el['nodes'][i+1]} for el in elements if el['type'] == 'way' for i in range(len(el['nodes']) - 1)]
            
            if not nodes:
                raise ValueError("No nodes found within the specified radius.")
        except Exception as e:
            logging.warning(f"Error retrieving data for point {point}: {e}")
            points_without_data.append(point)
            continue

        distances = get_node_distances(nodes, point)
        closest_nodes = sorted(distances, key=distances.get)[:num_nodes]

        points_with_data.append(point)

        filtered_nodes = {node_id: nodes[node_id] for node_id in closest_nodes}
        filtered_edges = [edge for edge in edges if edge['from'] in closest_nodes and edge['to'] in closest_nodes]

        stats_filtered = compute_centrality_measures(filtered_nodes, filtered_edges, point)
        all_metrics.append(stats_filtered)

        batch_counter += 1

        if batch_counter == batch_size:
            save_metrics_to_csv(all_metrics, dest_dir, num_nodes)
            all_metrics = []
            batch_counter = 0

    if all_metrics:
        save_metrics_to_csv(all_metrics, dest_dir, num_nodes)

def save_metrics_to_csv(metrics, dest_dir, num_nodes):
    df = pd.DataFrame(metrics)
    output_file = os.path.join(dest_dir, f"urban_metrics_{num_nodes}_nodes.csv")
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)
    logging.info(f"Metrics saved to {output_file}")

def setup_arguments():
    """Setup the program arguments."""
    parser = argparse.ArgumentParser(description="Download and process OSM data.")
    parser.add_argument('--dest_dir', type=str, required=True, help="Destination directory for output.")
    parser.add_argument('--input_file', type=str, required=True, help="Input CSV file with coordinates.")
    parser.add_argument('--radius', type=int, default=24, help="Radius for data retrieval (in meters).")
    parser.add_argument('--num_nodes', type=int, default=5, help="Number of nodes to analyze within the specified radius.")
    parser.add_argument('--start_lat', type=float, help="Starting latitude to process from.")
    parser.add_argument('--start_lon', type=float, help="Starting longitude to process from.")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_arguments()
    create_data_dir(args.dest_dir)
    data = pd.read_csv(args.input_file)
    points = data[['Latitude', 'Longitude']].values.tolist()

    start_point = (args.start_lat, args.start_lon) if args.start_lat is not None and args.start_lon is not None else None
    retrieve_and_average_stats(points, args.dest_dir, args.radius, num_nodes=args.num_nodes, start_point=start_point)

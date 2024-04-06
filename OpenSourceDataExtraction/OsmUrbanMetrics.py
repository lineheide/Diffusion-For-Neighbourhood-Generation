# OsmUrbanMetrics.py
# Overpass needed or OSM connection


import pandas as pd
import numpy as np
import osmnx as ox
import argparse
import os
from geopy.distance import great_circle
from tqdm import tqdm
import networkx as nx
import statistics as st

def create_data_dir(dir_path):
    """Create a directory to save the output."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {dir_path}: {e}")

def get_node_distances(G, center_point):
    """Calculate distances of all nodes in G from center_point."""
    distances = {}
    for node, data in G.nodes(data=True):
        node_point = (data['y'], data['x'])
        distance = great_circle(center_point, node_point).meters
        distances[node] = distance
    return distances

def retrieve_and_average_stats(points, dest_dir, radius, network_type='all', num_nodes=5):
    """Retrieve urban metrics for nodes within the specified radius, average the metrics, 
    and classify points based on data availability."""
    all_metrics = []
    points_with_data = []
    points_without_data = []

    for point in tqdm(points, desc="Processing Points"):
        try:
            G = ox.graph_from_point(point, dist=radius, simplify=True, network_type=network_type)
            if G.number_of_nodes() == 0:
                raise ValueError("No nodes found within the specified radius.")
        except Exception as e:
            print(f"Error retrieving graph for point {point}: {e}")
            points_without_data.append(point)
            continue

        distances = get_node_distances(G, point)
        closest_nodes = sorted(distances, key=distances.get)[:num_nodes]

        # Even if fewer nodes are found, proceed with those available
        points_with_data.append(point)

        G_sub = G.subgraph(closest_nodes)
        G_sub_directed = ox.get_digraph(G_sub)
        stats = ox.basic_stats(G_sub, area=radius**2 * np.pi)

        stats_filtered = {
            "Latitude": point[0],
            "Longitude": point[1],
            "n": len(G_sub.nodes()),
            "m": len(G_sub.edges()),
            "k_avg": stats.get("avg_degree", np.nan),
            "intersection_count": stats.get("intersection_count", np.nan),
            "streets_per_node_avg": stats.get("streets_per_node_avg", np.nan),
        }

        # Adding centrality measures to stats_filtered
        bc = nx.betweenness_centrality(G_sub_directed, weight="length")
        cc = nx.closeness_centrality(G_sub_directed, distance="length")
        dc = nx.degree_centrality(G_sub_directed)

        stats_filtered.update({
            "betweeness_centrality_min": min(bc.values()) if bc else np.nan,
            "betweeness_centrality_max": max(bc.values()) if bc else np.nan,
            "betweeness_centrality_avg": st.mean(bc.values()) if bc else np.nan,
            "betweeness_centrality_median": st.median(bc.values()) if bc else np.nan,
            "closeness_centrality_min": min(cc.values()) if cc else np.nan,
            "closeness_centrality_max": max(cc.values()) if cc else np.nan,
            "closeness_centrality_avg": st.mean(cc.values()) if cc else np.nan,
            "closeness_centrality_median": st.median(cc.values()) if cc else np.nan,
            "degree_centrality_min": min(dc.values()) if dc else np.nan,
            "degree_centrality_max": max(dc.values()) if dc else np.nan,
            "degree_centrality_avg": st.mean(dc.values()) if dc else np.nan,
            "degree_centrality_median": st.median(dc.values()) if dc else np.nan,
        })

        all_metrics.append(stats_filtered)

    # Save all metrics to CSV, as before
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        output_file = os.path.join(dest_dir, f"urban_metrics_{num_nodes}_nodes.csv")
        df.to_csv(output_file, index=False)
        print(f"Metrics saved to {output_file}")

    # Save points_with_data and points_without_data to CSV files
    df_with_data = pd.DataFrame(points_with_data, columns=['Latitude', 'Longitude'])
    df_without_data = pd.DataFrame(points_without_data, columns=['Latitude', 'Longitude'])

    with_data_file = os.path.join(dest_dir, "points_with_data.csv")
    without_data_file = os.path.join(dest_dir, "points_without_data.csv")

    df_with_data.to_csv(with_data_file, index=False)
    df_without_data.to_csv(without_data_file, index=False)

    print(f"Points with data saved to {with_data_file}")
    print(f"Points without data saved to {without_data_file}")

def setup_arguments():
    """Setup the program arguments."""
    parser = argparse.ArgumentParser(description="Download and process OSM data.")
    parser.add_argument('--dest_dir', type=str, required=True, help="Destination directory for output.")
    parser.add_argument('--input_file', type=str, required=True, help="Input CSV file with coordinates.")
    parser.add_argument('--radius', type=int, default=1000, help="Radius for data retrieval (in meters).")
    parser.add_argument('--num_nodes', type=int, default=5, help="Number of nodes to analyze within the specified radius.")
    parser.add_argument('--network', type=str, default='all', help="Type of network to download ('drive', 'walk', etc.).")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_arguments()
    create_data_dir(args.dest_dir)
    data = pd.read_csv(args.input_file)
    points = data[['Latitude', 'Longitude']].values.tolist()
    retrieve_and_average_stats(points, args.dest_dir, args.radius, network_type=args.network, num_nodes=args.num_nodes)

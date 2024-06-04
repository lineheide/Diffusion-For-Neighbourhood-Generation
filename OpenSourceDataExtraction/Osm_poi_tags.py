import pandas as pd
import requests
import argparse
import os
from tqdm import tqdm
from math import log, isnan, radians, cos
import torch
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_arguments():
    parser = argparse.ArgumentParser(description="Retrieve and categorize POIs from OSM around specified coordinates.")
    parser.add_argument('--input_file', type=str, required=True, help='CSV file with latitude and longitude.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output CSV files.')
    parser.add_argument('--radius', type=int, default=50, help='Search radius in meters.')
    return parser.parse_args()

def create_data_dir(dir_path):
    """Create a directory to save the output."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {e}")

def query_overpass(lat, lon, radius, max_retries=3, backoff_factor=1.0):
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
        [out:json][timeout:60];
        (
          node(around:{radius},{lat},{lon});
          way(around:{radius},{lat},{lon});
          rel(around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(overpass_url, params={'data': overpass_query})
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            logging.info(f"Overpass API response for coordinates ({lat}, {lon}): {data}")
            return data
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError for coordinates {lat}, {lon} on attempt {attempt + 1}: {e}")
        except requests.exceptions.RequestException as e:
            logging.error(f"RequestException for coordinates {lat}, {lon} on attempt {attempt + 1}: {e}")
        except ValueError as e:
            logging.error(f"ValueError (invalid JSON) for coordinates {lat}, {lon} on attempt {attempt + 1}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error for coordinates {lat}, {lon} on attempt {attempt + 1}: {e}")
        time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
    return None  # Return None if all retries failed

def categorize_and_count(elements, categories):
    counts = {category: 0 for category in categories.values()}
    for element in elements:
        tags = element.get('tags', {})
        for key, value in tags.items():
            category = categories.get(f"{key}:{value}")
            if category:
                counts[category] += 1
    return counts

def calculate_shannon_diversity_index(counts, device):
    total = sum(counts.values())
    if total == 0:
        return 0

    counts_tensor = torch.tensor([count for count in counts.values()], dtype=torch.float32, device=device)
    total_tensor = torch.tensor(total, dtype=torch.float32, device=device)
    proportions = counts_tensor / total_tensor
    log_proportions = torch.log(proportions)
    shannon_diversity = -torch.sum(proportions * log_proportions)

    return shannon_diversity.item() if not isnan(shannon_diversity.item()) else 0

def get_max_elevation(lat, lon, distance):
    directions = [(lat + distance / 111111, lon), # North
                  (lat - distance / 111111, lon), # South
                  (lat, lon + distance / (111111 * cos(radians(lat)))), # East
                  (lat, lon - distance / (111111 * cos(radians(lat))))] # West

    max_elevation = None
    for direction in directions:
        try:
            response = query_overpass(direction[0], direction[1], distance)
            if response:
                elements = response.get('elements', [])
                for element in elements:
                    elevation = float(element.get('tags', {}).get('ele', 0))
                    if max_elevation is None or elevation > max_elevation:
                        max_elevation = elevation
        except Exception as e:
            logging.warning(f"Error querying elevation for direction {direction}: {e}")
            continue

    return max_elevation

def process_coordinates(input_file, output_dir, radius, categories, device, batch_size=50):
    data = pd.read_csv(input_file)
    results = []
    columns = None

    # Generate output file path
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    for index, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {os.path.basename(input_file)}"):
        lat, lon = row['Latitude'], row['Longitude']
        try:
            response = query_overpass(lat, lon, radius)
            if response is None:  # Skip processing if no valid response
                logging.warning(f"No valid response for coordinates {lat}, {lon}")
                continue
            elements = response.get('elements', [])
            if not elements:
                logging.warning(f"No elements found for coordinates {lat}, {lon}")
                continue

            counts = categorize_and_count(elements, categories)
            
            # Separate urban vibrancy counts and other counts
            urban_vibrancy_counts = {k: v for k, v in counts.items() if k.startswith('poi_')}
            other_counts = {k: v for k, v in counts.items() if not k.startswith('poi_')}
            
            diversity_index = calculate_shannon_diversity_index(urban_vibrancy_counts, device)

            # Calculate slope
            current_elevation_data = response.get('elements', [])
            current_elevation = float(current_elevation_data[0].get('tags', {}).get('ele', 0)) if current_elevation_data else 0
            max_elevation = get_max_elevation(lat, lon, 80)  # Using 80 meters as the distance
            slope = (max_elevation - current_elevation) if max_elevation else 0

            # Combine results with appropriate columns
            combined_counts = {**other_counts, **urban_vibrancy_counts}
            combined_counts['Latitude'] = lat
            combined_counts['Longitude'] = lon
            combined_counts['Diversity_Index'] = diversity_index
            combined_counts['Slope'] = slope
            results.append(combined_counts)

            # Write to CSV in batches
            if (index + 1) % batch_size == 20 or (index + 1) == len(data):
                results_df = pd.DataFrame(results)
                if columns is None:
                    columns = ['Latitude', 'Longitude'] + [col for col in results_df.columns if col not in ['Latitude', 'Longitude', 'Diversity_Index', 'Slope']]
                results_df = results_df[columns + ['Diversity_Index', 'Slope']]
                if os.path.exists(output_file):
                    results_df.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    results_df.to_csv(output_file, mode='w', header=True, index=False)
                results = []

        except Exception as e:
            logging.warning(f"Error processing coordinates {lat}, {lon}: {e}")

    if results:
        results_df = pd.DataFrame(results)
        if columns is None:
            columns = ['Latitude', 'Longitude'] + [col for col in results_df.columns if col not in ['Latitude', 'Longitude', 'Diversity_Index', 'Slope']]
        results_df = results_df[columns + ['Diversity_Index', 'Slope']]
        if os.path.exists(output_file):
            results_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(output_file, mode='w', header=True, index=False)
    print(f"Data from {input_file} has been processed and output to {output_file}")

if __name__ == "__main__":
    args = setup_arguments()
    create_data_dir(args.output_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    categories = {
        ################################################ These are the POIs, that are used for urban vibrancy calulcation ################################################
        # Urban Vibrancy
        "amenity:arts_centre": "poi_amenity",
        "amenity:community_centre": "poi_amenity",
        "amenity:fountain": "poi_amenity",
        "amenity:library": "poi_amenity",
        "amenity:marketplace": "poi_amenity",
        "amenity:place_of_worship": "poi_amenity",
        "amenity:public_bookcase": "poi_amenity",
        "amenity:social_centre": "poi_amenity",
        "amenity:theatre": "poi_amenity",
        "amenity:townhall": "poi_amenity",
        "amenity:atm": "poi_amenity",
        "amenity:bank": "poi_amenity",
        "amenity:bar": "poi_amenity",
        "amenity:bicycle_rental": "poi_amenity",
        "amenity:biergarten": "poi_amenity",
        "amenity:buddhist": "poi_amenity",
        "amenity:cafe": "poi_amenity",
        "amenity:car_rental": "poi_amenity",
        "amenity:car_wash": "poi_amenity",
        "amenity:christian": "poi_amenity",
        "amenity:cinema": "poi_amenity",
        "amenity:clinic": "poi_amenity",
        "amenity:college": "poi_amenity",
        "amenity:courthouse": "poi_amenity",
        "amenity:dentist": "poi_amenity",
        "amenity:doctors": "poi_amenity",
        "amenity:fast_food": "poi_amenity",
        "amenity:fire_station": "poi_amenity",
        "amenity:food_court": "poi_amenity",
        "amenity:graveyard": "poi_amenity",
        "amenity:hospital": "poi_amenity",
        "amenity:kindergarten": "poi_amenity",
        "amenity:market_place": "poi_amenity",
        "amenity:nightclub": "poi_amenity",
        "amenity:pharmacy": "poi_amenity",
        "amenity:police": "poi_amenity",
        "amenity:post_office": "poi_amenity",
        "amenity:pub": "poi_amenity",
        "amenity:public_building": "poi_amenity",
        "amenity:restaurant": "poi_amenity",
        "amenity:school": "poi_amenity",
        "amenity:toilet": "poi_amenity",
        "amenity:university": "poi_amenity",
        "amenity:veterinary": "poi_amenity",
        "shop:bakery": "poi_shop",
        "shop:beauty_shop": "poi_shop",
        "shop:beverages": "poi_shop",
        "shop:bicycle_shop": "poi_shop",
        "shop:bookshop": "poi_shop",
        "shop:butcher": "poi_shop",
        "shop:car_dealership": "poi_shop",
        "shop:chemist": "poi_shop",
        "shop:clothes": "poi_shop",
        "shop:computer_shop": "poi_shop",
        "shop:convenience": "poi_shop",
        "shop:department_store": "poi_shop",
        "shop:florist": "poi_shop",
        "shop:furniture_shop": "poi_shop",
        "shop:garden_centre": "poi_shop",
        "shop:gift_shop": "poi_shop",
        "shop:greengrocer": "poi_shop",
        "shop:hairdresser": "poi_shop",
        "shop:jeweller": "poi_shop",
        "shop:kiosk": "poi_shop",
        "shop:laundry": "poi_shop",
        "shop:mall": "poi_shop",
        "shop:mobile_phone_shop": "poi_shop",
        "shop:newsagent": "poi_shop",
        "shop:optician": "poi_shop",
        "shop:outdoor_shop": "poi_shop",
        "shop:shoe_shop": "poi_shop",
        "shop:sports_shop": "poi_shop",
        "shop:stationery": "poi_shop",
        "shop:supermarket": "poi_shop",
        "shop:toy_shop": "poi_shop",
        "shop:travel_agent": "poi_shop",
        "amenity:bar": "poi_food",
        "amenity:cafe": "poi_food",
        "amenity:fast_food": "poi_food",
        "amenity:pub": "poi_food",
        "amenity:restaurant": "poi_food",
        "amenity:bicycle_parking": "poi_transportation",
        "amenity:bus_station": "poi_transportation",
        "amenity:car_rental": "poi_transportation",
        "amenity:charging_station": "poi_transportation",
        "amenity:ferry_terminal": "poi_transportation",
        "amenity:fuel": "poi_transportation",
        "amenity:parking": "poi_transportation",
        "amenity:taxi": "poi_transportation",
        "leisure:garden": "poi_leisure",
        "leisure:nature_reserve": "poi_leisure",
        "leisure:park": "poi_leisure",
        "leisure:stadium": "poi_leisure",
        "leisure:playground": "poi_leisure",
        "leisure:dog_park": "poi_leisure",
        "leisure:pitch": "poi_leisure",
        "leisure:sports_centre": "poi_leisure",
        "leisure:track": "poi_leisure",
        "tourism:artwork": "poi_tourism",
        "tourism:attraction": "poi_tourism",
        "tourism:gallery": "poi_tourism",
        "tourism:museum": "poi_tourism",
        "tourism:viewpoint": "poi_tourism",
        "tourism:zoo": "poi_tourism",
        "natural:tree": "poi_natural",
        "natural:waterfall": "poi_natural",
        "natural:beach": "poi_natural",
        "natural:peak": "poi_natural",
        "building:apartments": "poi_buiding",
        "building:commercial": "poi_buiding",
        "building:hotel": "poi_buiding",
        "building:industrial": "poi_buiding",
        "building:retail": "poi_buiding",
        "building:house": "poi_buiding",
        "historic:castle": "poi_historic",
        "historic:memorial": "poi_historic",
        "historic:monument": "poi_historic",
        "historic:statue": "poi_historic",
        "historic:ruins": "poi_historic",
        "historic:archaeological_site": "poi_historic",
        "tourism:guesthouse": "poi_historic",
        "tourism:hostel": "poi_historic",
        "tourism:hotel": "poi_historic",
        "tourism:picnic_site": "poi_historic",
        "tourism:theme_park": "poi_historic",
        "tourism:information": "poi_historic",
        "tourism:viewpoint": "poi_historic",
        "historic:wayside_shrine": "poi_historic",
        "highway:footway": "poi_highway",
        "highway:cycleway": "poi_highway",
        "highway:living_street": "poi_highway",
        "highway:pedestrian": "poi_highway",
        "highway:residential": "poi_highway",
        "highway:steps": "poi_highway",
        "highway:track": "poi_highway",
        ################################################ Lines Tags for Catgeories ################################################
        # Food and Drink
        "amenity:restaurant": "food_drink",
        "amenity:cafe": "food_drink",
        "amenity:bar": "food_drink",
        "amenity:pub": "food_drink",
        "amenity:biergarten": "food_drink",
        "amenity:fast_food": "food_drink",
        "amenity:food_court": "food_drink",
        "amenity:ice_cream": "food_drink",
        "amenity:marketplace": "food_drink",
        "shop:supermarket": "food_drink",
        "shop:department_store": "food_drink",
        "shop:mall": "food_drink",
        "shop:clothes": "food_drink",
        "shop:alcohol": "food_drink",
        "shop:bakery": "food_drink",
        "shop:beverages": "food_drink",
        "shop:brewing_supplies": "food_drink",
        "shop:butcher": "food_drink",
        "shop:cheese": "food_drink",
        "shop:chocolate": "food_drink",
        "shop:coffee": "food_drink",
        "shop:confectionery": "food_drink",
        "shop:convenience": "food_drink",
        "shop:deli": "food_drink",
        "shop:dairy": "food_drink",
        "shop:farm": "food_drink",
        "shop:frozen_food": "food_drink",
        "shop:greengrocer": "food_drink",
        "shop:health_food": "food_drink",
        "shop:ice_cream": "food_drink",
        "shop:organic": "food_drink",
        "shop:pasta": "food_drink",
        "shop:pastry": "food_drink",
        "shop:seafood": "food_drink",
        "shop:spices": "food_drink",
        "shop:tea": "food_drink",
        "shop:wine": "food_drink",
        "shop:water": "food_drink",
        "shop:department_store": "food_drink",
        "shop:general": "food_drink",
        "shop:kiosk": "food_drink",
        # Buildings and Facilities
        "building:commercial": "buildings_facilities",
        "building:residential": "buildings_facilities",
        "building:house": "buildings_facilities",
        "building:apartments": "buildings_facilities",
        "building:retail": "buildings_facilities",
        "building:industrial": "buildings_facilities",
        "building:warehouse": "buildings_facilities",
        "building:school": "buildings_facilities",
        "building:university": "buildings_facilities",
        "building:college": "buildings_facilities",
        "building:kindergarten": "buildings_facilities",
        "amenity:hospital": "buildings_facilities",
        "amenity:police": "buildings_facilities",
        "amenity:fire_station": "buildings_facilities",
        "amenity:library": "buildings_facilities",
        "amenity:community_centre": "buildings_facilities",
        "amenity:school": "buildings_facilities",
        "amenity:university": "buildings_facilities",
        "amenity:college": "buildings_facilities",
        "amenity:kindergarten": "buildings_facilities",
        "amenity:clinic": "buildings_facilities",
        "amenity:doctors": "buildings_facilities",
        "amenity:dentist": "buildings_facilities",
        "amenity:veterinary": "buildings_facilities",
        "leisure:pitch": "buildings_facilities",
        "leisure:stadium": "buildings_facilities",
        "leisure:sports_centre": "buildings_facilities",
        "leisure:swimming_pool": "buildings_facilities",
        "office:it": "buildings_facilities",
        "office:software": "buildings_facilities",
        "office:research": "buildings_facilities",
        "amenity:bureau_de_change": "buildings_facilities",
        "amenity:money_transfer": "buildings_facilities",
        "amenity:shelter": "buildings_facilities",
        "amenity:social_facility": "buildings_facilities",
        "amenity:social_centre": "buildings_facilities",
        # Roads and Transportation
        "highway:motorway": "roads_transportation",
        "highway:motorway_link": "roads_transportation",
        "highway:trunk": "roads_transportation",
        "highway:trunk_link": "roads_transportation",
        "highway:primary": "roads_transportation",
        "highway:primary_link": "roads_transportation",
        "highway:secondary": "roads_transportation",
        "highway:secondary_link": "roads_transportation",
        "highway:tertiary": "roads_transportation",
        "highway:tertiary_link": "roads_transportation",
        "highway:residential": "roads_transportation",
        "highway:living_street": "roads_transportation",
        "highway:service": "roads_transportation",
        "highway:unclassified": "roads_transportation",
        "highway:road": "roads_transportation",
        "highway:footway": "roads_transportation",
        "highway:pedestrian": "roads_transportation",
        "highway:path": "roads_transportation",
        "highway:cycleway": "roads_transportation",
        "highway:bridleway": "roads_transportation",
        "highway:steps": "roads_transportation",
        "highway:track": "roads_transportation",
        "highway:busway": "roads_transportation",
        "highway:construction": "roads_transportation",
        "highway:proposed": "roads_transportation",
        "highway:escape": "roads_transportation",
        "highway:emergency_bay": "roads_transportation",
        "highway:crossing": "roads_transportation",
        "highway:turning_circle": "roads_transportation",
        "highway:turning_loop": "roads_transportation",
        "highway:mini_roundabout": "roads_transportation",
        "highway:traffic_signals": "roads_transportation",
        "highway:stop": "roads_transportation",
        "highway:give_way": "roads_transportation",
        "highway:services": "roads_transportation",
        "highway:rest_area": "roads_transportation",
        "highway:speed_camera": "roads_transportation",
        "highway:toll_booth": "roads_transportation",
        "highway:milestone": "roads_transportation",
        "highway:passing_place": "roads_transportation",
        "highway:bus_stop": "roads_transportation",
        "highway:street_lamp": "roads_transportation",
        # Tourism and Leisure
        "tourism:hotel": "roads_transportation",
        "tourism:motel": "tourism_leisure",
        "tourism:guest_house": "tourism_leisure",
        "tourism:hostel": "tourism_leisure",
        "tourism:bed_and_breakfast": "tourism_leisure",
        "tourism:camp_site": "tourism_leisure",
        "tourism:caravan_site": "roads_transportation",
        "tourism:chalet": "roads_transportation",
        "tourism:information": "roads_transportation",
        "tourism:museum": "roads_transportation",
        "tourism:gallery": "roads_transportation",
        "tourism:zoo": "roads_transportation",
        "tourism:theme_park": "roads_transportation",
        "tourism:attraction": "roads_transportation",
        "tourism:viewpoint": "roads_transportation",
        "tourism:picnic_site": "roads_transportation",
        "leisure:park": "roads_transportation",
        "leisure:garden": "roads_transportation",
        "leisure:nature_reserve": "roads_transportation",
        "leisure:bowling_alley": "roads_transportation",
        "leisure:arcade": "roads_transportation",
        "leisure:playground": "roads_transportation",
        "amenity:cinema": "roads_transportation",
        "amenity:gym": "roads_transportation",
        "amenity:theatre": "roads_transportation",
        "amenity:arts_centre": "roads_transportation",
        "amenity:studio": "roads_transportation",
        "amenity:nightclub": "roads_transportation",
        "amenity:casino": "roads_transportation",
        "amenity:stripclub": "roads_transportation",
        "amenity:event_venue": "roads_transportation",
        "tourism:festival": "roads_transportation",
        "tourism:party_venue": "roads_transportation",
        # Greenery and Natural Features
        "landuse:forest": "greenery_natural",
        "landuse:grass": "greenery_natural",
        "landuse:orchard": "greenery_natural",
        "landuse:farmland": "greenery_natural",
        "landuse:allotments": "greenery_natural",
        "landuse:recreation_ground": "greenery_natural",
        "landuse:vineyard": "greenery_natural",
        "natural:wood": "greenery_natural",
        "natural:tree": "greenery_natural",
        "natural:tree_row": "greenery_natural",
        "natural:scrub": "greenery_natural",
        "natural:heath": "greenery_natural",
        "natural:grassland": "greenery_natural",
        "natural:moor": "greenery_natural",
        "natural:wetland": "greenery_natural",
        "natural:mud": "greenery_natural",
        "natural:land": "greenery_natural",
        "amenity:dog_park": "greenery_natural",
        "amenity:picnic_site": "greenery_natural",
        "amenity:park": "greenery_natural",
        "amenity:garden": "greenery_natural",
        "amenity:nature_reserve": "greenery_natural",
        # Public Services and Facilities
        "amenity:place_of_worship": "public_services",
        "amenity:post_office": "public_services",
        "amenity:bank": "public_services",
        "amenity:atm": "public_services",
        "amenity:pharmacy": "public_services",
        "amenity:prison": "public_services",
        "amenity:lawyer": "public_services",
        "amenity:notary": "public_services",
        "amenity:townhall": "public_services",
        "amenity:courthouse": "public_services",
        "amenity:embassy": "public_services",
        "amenity:government": "public_services",
        "amenity:telephone": "public_services",
        "amenity:internet_cafe": "public_services",
        "amenity:newsagent": "public_services",
        "amenity:waste_basket": "public_services",
        "amenity:waste_disposal": "public_services",
        "amenity:waste_transfer_station": "public_services",
        "amenity:water_fountain": "public_services",
        # Arts, Entertainment, and Events
        "amenity:studio": "arts_entertainment_events",
        "amenity:arts_centre": "arts_entertainment_events",
        "amenity:nightclub": "arts_entertainment_events",
        "amenity:casino": "arts_entertainment_events",
        "amenity:stripclub": "arts_entertainment_events",
        "amenity:event_venue": "arts_entertainment_events",
        "tourism:festival": "arts_entertainment_events",
        "tourism:party_venue": "arts_entertainment_events",
        # Beauty, Personal Care, and Health
        "amenity:beauty_salon": "beauty_personal_health",
        "amenity:hairdresser": "beauty_personal_health",
        "amenity:nail_salon": "beauty_personal_health",
        "amenity:clinic": "beauty_personal_health",
        "amenity:doctors": "beauty_personal_health",
        "amenity:dentist": "beauty_personal_health",
        "amenity:veterinary": "beauty_personal_health",
        # Miscellaneous and Other Services
        "amenity:social_facility": "miscellaneous_services",
        "amenity:social_centre": "miscellaneous_services",
        "amenity:coworking_space": "miscellaneous_services",
        "office:it": "miscellaneous_services",
        "office:software": "miscellaneous_services",
        "office:research": "miscellaneous_services",
        "amenity:bureau_de_change": "miscellaneous_services",
        "amenity:money_transfer": "miscellaneous_services",
        "amenity:shelter": "miscellaneous_services",
        "amenity:recycling": "miscellaneous_services",
        "amenity:theatre": "miscellaneous_services",
        "amenity:newsagent": "miscellaneous_services",
        # Water Bodies
        "natural:water": "water_body",
        "natural:lake": "water_body",
        "natural:river": "water_body",
        "natural:stream": "water_body",
        "natural:pond": "water_body",
        "waterway:river": "water_body",
        "waterway:stream": "water_body",
        "waterway:canal": "water_body",
        "waterway:drain": "water_body",
        "waterway:ditch": "water_body"
    }

    process_coordinates(args.input_file, args.output_dir, args.radius, categories, device)

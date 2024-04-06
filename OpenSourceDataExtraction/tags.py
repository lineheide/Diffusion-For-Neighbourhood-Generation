def create_osm_tags():
    # Places for social, cultural, and community activities
    amenity_tags = ["arts_centre", "community_centre", "fountain", "library", "marketplace", "place_of_worship",
                    "public_bookcase", "social_centre", "theatre", "townhall", "arts_centre", "atm", "bank",
                    "bar", "bicycle_rental", "biergarten", "buddhist", "cafe", "car_rental", "car_wash",
                    "christian", "christian_anglican", "christian_catholic", "christian_evangelical",
                    "christian_lutheran", "christian_orthodox", "cinema", "clinic", "college", "community_centre",
                    "courthouse", "dentist", "doctors", "fast_food", "fire_station", "food_court", "fountain",
                    "graveyard", "hindu", "hospital", "jewish", "kindergarten", "library", "market_place", "muslim",
                    "nightclub", "pharmacy", "police", "post_office", "pub", "public_building", "restaurant", "school",
                    "theatre", "toilet", "town_hall", "university", "vending_any", "vending_machine", "veterinary"]
    
    # Shopping and commercial services
    shop_tags = ["bakery", "beauty_shop", "beverages", "bicycle_shop", "bookshop", "butcher", "car_dealership", "chemist", "clothes", 
                 "computer_shop", "convenience", "department_store", "florist", "furniture_shop", "garden_centre", "gift_shop",
                 "greengrocer", "hairdresser", "jeweller", "kiosk", "laundry", "mall", "mobile_phone_shop", "newsagent", "optician",
                 "outdoor_shop", "shoe_shop", "sports_shop", "stationery", "supermarket", "toy_shop", "travel_agent", "bakery", "bicycle",
                 "bookshop", "clothes", "convenience", "department_store", "electronics", "florist", "gift", "supermarket"]
    
    # Food and drink establishments
    food_tags = ["bar", "cafe", "fast_food", "pub", "restaurant"]
    
    # Infrastructure and transport facilities
    transportation_tags = ["bicycle_parking", "bus_station", "car_rental", "charging_station", "ferry_terminal",
                           "fuel", "parking", "taxi"]
    
    # Leisure and entertainment
    leisure_tags = ["garden", "nature_reserve", "park", "stadium", "playground", "dog_park", "park", "pitch", "playground", "sports_centre", "stadium", "track"]
    
    # Sightseeing and tourism
    tourism_tags = ["artwork", "attraction", "gallery", "museum", "viewpoint", "zoo"]
    
    # Green spaces and natural features
    natural_tags = ["tree", "waterfall", "beach", "peak"]
    
    # Buildings and historical sites
    building_tags = ["apartments", "commercial", "hotel", "industrial", "retail", "house"]
    historic_tags = ["castle", "memorial", "monument", "statue", "ruins", "archaeological", "artwork", "attraction",
                     "castle", "guesthouse", "hostel", "hotel", "memorial", "monument", "museum", "picnic_site",
                     "theme_park", "tourist_info", "viewpoint", "wayside_shrine", "zoo"]

    # Urban infrastructure
    highway_tags = ["footway", "cycleway", "living_street", "pedestrian", "residential", "steps"]

    
    osm_tags = {
        "amenity": amenity_tags,
        "shop": shop_tags,
        "food": food_tags,
        "transportation": transportation_tags,
        "leisure": leisure_tags,
        "tourism": tourism_tags,
        "natural": natural_tags,
        "building": building_tags,
        "historic": historic_tags,
        "highway": highway_tags
    }

    return osm_tags

if __name__ == "__main__":
    osm_tags = create_osm_tags()
    print(osm_tags)

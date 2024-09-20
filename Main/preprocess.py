import os
import json
import pandas as pd

# files_directory = 'raw_ratings/'

def preprocess_bike_ratings(file_path, user):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed_data = []

    for file_path, rating in data.items():
        bike_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Convert rating to integer
        if rating == "usable":
            rating = 1
        elif rating == "not_usable":
            rating = 0
        else:
            continue  # Skip invalid ratings
        
        processed_data.append({
            "bid": bike_id,
            "rating": rating,
            "user": os.path.splitext(os.path.basename(user))[0],
        })
    
    return processed_data

def preprocess_ratings(files_directory, save_file = 0):
    all_processed_data = []
    for file_name in os.listdir(files_directory):
        if file_name.endswith('.txt'):  
            file_path = os.path.join(files_directory, file_name)
            processed_data = preprocess_bike_ratings(file_path, file_name)
            all_processed_data.extend(processed_data)


    df = pd.DataFrame(all_processed_data)

    if save_file:
        df.to_csv('processed_ratings.csv', index=False)

    return df
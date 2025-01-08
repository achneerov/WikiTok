import os
import json

def find_json_by_id(folder_path, target_id):
    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    
                    # Iterate over each dictionary in the list
                    for entry in data:
                        if isinstance(entry, dict) and entry.get('id') == target_id:
                            print(json.dumps(entry, indent=4))
                            return  # Exit once the target ID is found
                except json.JSONDecodeError:
                    print(f"Error reading {filename}")
    
    print("ID not found in any file.")

# Example usage
folder_path = 'json'  # Replace with your folder path
target_id = "3100"  # Replace with the ID you're searching for
find_json_by_id(folder_path, target_id)

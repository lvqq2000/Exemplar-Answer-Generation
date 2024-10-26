import json

def read_json(file_path):
    """
    Reads JSON data from a specified file path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or None: The parsed JSON data as a dictionary, or None if an error occurs.
    """
    print(f"Reading from {file_path}...")
    try:
        # Open and load JSON data from file
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Handle file not found error
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    
    # Handle permission error when accessing the file
    except PermissionError:
        print(f"Error: Permission denied when trying to open '{file_path}'.")
        return None
    
    # Handle cases where the JSON data is invalid
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return None
    
    # Handle any other unexpected exceptions
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def save_df_to_json(df, output_file_path):
    """
    Saves a DataFrame to a specified file path as a JSON file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved as JSON.
        output_file_path (str): Path where the JSON file will be saved.

    Returns:
        None
    """
    try:
        # Convert DataFrame to JSON format and save to file
        df.to_json(output_file_path, orient='records', indent=4)
        print(f"Data successfully saved to {output_file_path}")
    
    # Handle cases where the specified directory does not exist
    except FileNotFoundError:
        print(f"Error: The directory for '{output_file_path}' was not found.")
    
    # Handle permission error when writing to the file
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{output_file_path}'.")
    
    # Handle ValueErrors, e.g., if the DataFrame is empty or incompatible with JSON format
    except ValueError as e:
        print(f"Error: {e}. Please ensure the DataFrame is valid for JSON conversion.")
    
    # Handle any other unexpected exceptions
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
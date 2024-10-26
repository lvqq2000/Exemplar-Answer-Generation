import json

def read_json(file_path):
    print(f"Reading from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied when trying to open '{file_path}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def save_df_to_json(df, output_file_path):
    try:
        df.to_json(output_file_path, orient='records', indent=4)
        print(f"Data successfully saved to {output_file_path}")
    except FileNotFoundError:
        print(f"Error: The directory for '{output_file_path}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to '{output_file_path}'.")
    except ValueError as e:
        print(f"Error: {e}. Please ensure the DataFrame is valid for JSON conversion.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
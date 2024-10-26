import pandas as pd
import json
import re
#from sklearn.model_selection import KFold
from config import TASK_CONTENT_MAX_LENGTH, QUESTION_MAX_LENGTH, RUBRIC_MAX_LENGTH

def preprocess_dataframe(df):
    df = fill_missing_task_info(df)

    df['task_content'] = df['task_content'].apply(clean_text)

    df = remove_long_entries(df, 'task_content', length_limit=TASK_CONTENT_MAX_LENGTH)
    df = remove_long_entries(df, 'question', length_limit=QUESTION_MAX_LENGTH)
    df = remove_long_entries(df, 'rubric', length_limit=RUBRIC_MAX_LENGTH)

    # Apply the function to the rubric column and create new columns
    df[['indicator', 'criteria', 'curriculum_codes']] = df['rubric'].apply(extract_rubric_info)

    # Convert curriculum_codes column to strings
    df['curriculum_codes'] = df['curriculum_codes'].apply(lambda x: json.dumps(x, sort_keys=True))

    # Drop the original rubric column if no longer needed
    df.drop(columns=['rubric'], inplace=True)

    df = df.dropna()

    df = df.drop_duplicates()

    # return df.to_dict(orient='records')  # Convert back to list of dicts
    return df

def clean_text(text):
    # Remove all HTML entities and tags
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r'<[^>]*>', ' ', text)  # Remove HTML tags

    # Remove specific punctuation characters
    text = text.replace(';', '').replace('!', '').replace('?', '')

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

# Function to extract the desired values from the rubric
def extract_rubric_info(rubric):
    rubric_data = json.loads(rubric)
    indicator = rubric_data['items'][0]
    criteria = rubric_data['criteria']
    curriculum_codes = rubric_data['curriculum_codes']
    return pd.Series([indicator, criteria, curriculum_codes])

def fill_missing_task_info(df):
    if isinstance(df, list):
        df = pd.DataFrame(df)  # Convert the list to a DataFrame

    task_info = {}

    # Iterate over the DataFrame to populate the task_info dictionary
    for index, row in df.iterrows():
        task_id = row['task_id']
        
        # Store task_title and task_content for each task_id
        if task_id not in task_info:
            task_info[task_id] = {
                'task_title': row.get('task_title'),
                'task_content': row.get('task_content')
            }
        else:
            # Update existing task_info with non-null values
            if pd.isna(task_info[task_id]['task_title']):
                task_info[task_id]['task_title'] = row.get('task_title')
            if pd.isna(task_info[task_id]['task_content']):
                task_info[task_id]['task_content'] = row.get('task_content')

    # Fill missing values in the original DataFrame
    for index, row in df.iterrows():
        task_id = row['task_id']
        if task_id in task_info:
            if pd.isna(row['task_title']):
                df.at[index, 'task_title'] = task_info[task_id]['task_title']
            if pd.isna(row['task_content']):
                df.at[index, 'task_content'] = task_info[task_id]['task_content']

    return df

def remove_long_entries(df, column_name, length_limit=10000):
    """
    Remove rows from the DataFrame where the specified column exceeds the given length limit.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to check for length.
        length_limit (int): The maximum allowed length for the specified column.

    Returns:
        pd.DataFrame: The updated DataFrame with long entries removed.
    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Identify rows to be removed
    to_remove = df[df[column_name].str.len() > length_limit]

    # Print out entries being removed
    for index, row in to_remove.iterrows():
        print(f"Row with ID {row['question_id']} has been neglected since its '{column_name}' exceeds the length limit of {length_limit} characters.")

    # Remove rows exceeding the length limit
    new_df = df[df[column_name].str.len() <= length_limit]
    
    return new_df

def show_progress_bar(iteration, total, length=50):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r|{bar}| {percent:.1f}% Complete', end='\r')
    if iteration == total:
        print()  # New line at the end of the progress

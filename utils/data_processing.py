import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find

from config import QUESTION_MAX_LENGTH, RUBRIC_MAX_LENGTH, NLTK_RESOURCES, QUESTION_CATEGORY_NAME
from utils.question import categorize_question

# Download necessary NLTK resources
def download_nltk_resources():
    resources = NLTK_RESOURCES

    for resource in resources:
        try:
            find(f'corpora/{resource}.zip')  # Check for corpus resources
        except LookupError:
            nltk.download(resource)

# Call the function to ensure resources are available
download_nltk_resources()

def preprocess_dataframe(df):
    new_df = df.copy()

    new_df = fill_missing_task_info(new_df)

    new_df[QUESTION_CATEGORY_NAME] = new_df['question'].apply(categorize_question)

    new_df = remove_long_entries(new_df, 'question', length_limit=QUESTION_MAX_LENGTH)
    new_df = remove_long_entries(new_df, 'rubric', length_limit=RUBRIC_MAX_LENGTH)

    # Apply the function to the rubric column and create new columns
    new_df[['indicator', 'criteria', 'curriculum_codes']] = new_df['rubric'].apply(extract_rubric_info)

    # Convert curriculum_codes column to strings
    new_df['curriculum_codes'] = new_df['curriculum_codes'].apply(lambda x: json.dumps(x, sort_keys=True))

    # Drop the original rubric column if no longer needed
    new_df.drop(columns=['rubric'], inplace=True)

    new_df = new_df.dropna()

    new_df = new_df.drop_duplicates()

    return new_df

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

def preprocess_text(text):
    """Preprocesses the input text by lowercasing, removing punctuation,
       tokenizing, removing stopwords, and lemmatizing."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

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

    # Check if there are any rows to be removed
    if not to_remove.empty:
        # Collect question IDs to print
        neglected_ids = to_remove['question_id'].tolist()
        neglected_count = len(neglected_ids)

        # Print out the summary of entries being removed
        print(f"{neglected_count} files have been neglected since their '{column_name}' exceeds the length limit of {length_limit} characters. Question IDs: {', '.join(neglected_ids)}")

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

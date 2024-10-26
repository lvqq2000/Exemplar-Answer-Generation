import pandas as pd
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils.question import categorize_question
from config import QUESTION_CATEGORY_NAME, QUESTION_MAX_LENGTH, RUBRIC_MAX_LENGTH

def preprocess_dataframe(df, do_print=True):
    """
    Preprocess the DataFrame by filling missing task information, categorizing questions, 
    limiting entry lengths, extracting rubric components, and removing duplicates.
    """
    new_df = df.copy()
    new_df = fill_missing_task_info(new_df)

    new_df[QUESTION_CATEGORY_NAME] = new_df['question'].apply(categorize_question)

    # Remove long entries in 'question' and 'rubric' columns
    new_df = remove_long_entries(new_df, 'question', length_limit=QUESTION_MAX_LENGTH, do_print=do_print)
    new_df = remove_long_entries(new_df, 'rubric', length_limit=RUBRIC_MAX_LENGTH, do_print=do_print)

    # Extract rubric components into separate columns
    new_df[['indicator', 'criteria', 'curriculum_codes']] = new_df['rubric'].apply(extract_rubric_info)
    new_df['curriculum_codes'] = new_df['curriculum_codes'].apply(lambda x: json.dumps(x, sort_keys=True))

    new_df.drop(columns=['rubric'], inplace=True)
    new_df.dropna(inplace=True)
    new_df.drop_duplicates(inplace=True)

    return new_df

def clean_text(text):
    """
    Cleans text by removing HTML entities, tags, specific punctuation, and extra spaces.
    """
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[;!?]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation, tokenizing, 
    removing stopwords, and lemmatizing.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

def extract_rubric_info(rubric):
    """
    Extracts rubric details into indicator, criteria, and curriculum codes.
    """
    rubric_data = json.loads(rubric)
    return pd.Series([rubric_data['items'][0], rubric_data['criteria'], rubric_data['curriculum_codes']])

def fill_missing_task_info(df):
    """
    Fills missing task information by carrying forward task title and content based on task ID.
    """
    task_info = {}
    for index, row in df.iterrows():
        task_id = row['task_id']
        task_info.setdefault(task_id, {'task_title': row.get('task_title'), 'task_content': row.get('task_content')})
        task_info[task_id]['task_title'] = task_info[task_id].get('task_title') or row.get('task_title')
        task_info[task_id]['task_content'] = task_info[task_id].get('task_content') or row.get('task_content')

    for index, row in df.iterrows():
        task_id = row['task_id']
        if task_id in task_info:
            df.at[index, 'task_title'] = df.at[index, 'task_title'] or task_info[task_id]['task_title']
            df.at[index, 'task_content'] = df.at[index, 'task_content'] or task_info[task_id]['task_content']

    return df

def remove_long_entries(df, column_name, length_limit=10000, do_print=True):
    """
    Removes entries from DataFrame where column length exceeds length_limit.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    to_remove = df[df[column_name].str.len() > length_limit]
    if not to_remove.empty and do_print:
        neglected_ids = to_remove['question_id'].tolist()
        print(f"{len(neglected_ids)} entries removed; '{column_name}' exceeds {length_limit} characters. IDs: {', '.join(neglected_ids)}")

    return df[df[column_name].str.len() <= length_limit]

def show_progress_bar(iteration, total, length=50):
    """
    Displays a progress bar in the console.
    """
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r|{bar}| {percent:.1f}% Complete', end='\r')
    if iteration == total:
        print()  # New line at the end of the progress
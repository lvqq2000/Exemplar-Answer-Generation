import pandas as pd
import re
#from sklearn.model_selection import KFold

def preprocess_data(data):
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Drop unnecessary columns
    df.drop(columns=['question_id', 'task_id', 'task_title'], inplace=True)

    # Drop rows where the length of "task_content" is greater than 10,000
    df = df[df['task_content'].str.len() <= 10000]

    df['task_content'] = df['task_content'].apply(clean_text)

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
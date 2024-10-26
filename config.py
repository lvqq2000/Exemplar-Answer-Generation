import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# File paths
TRAIN_DATA_PATH = 'data/training_data.json'
VAL_SET_PATH = 'data/val_set.jsonl'
INPUT_DEFAULT_PATH = 'data/input.json'
OUTPUT_PATH = 'output/output.json'
VALIDATION_RESULT_PATH = 'output/validation_results.json'

# Input setting
TASK_CONTENT_MAX_LENGTH = 8000
QUESTION_MAX_LENGTH = 600
RUBRIC_MAX_LENGTH = 1000
EXPECTED_ROW_NAME = "answer"

# Output setting
OUTPUT_ROW_NAME = "examplar_answer"
OUTPUT_MAX_LENGTH = 700

# Model settings
MODEL_NAME = 'gpt-4o-mini'
SYSTEM_PROMPT = {
    "role": "system", 
    "content": 
    f"""
    You will be provided with a task content, a question related to task content and the rubric which includes indicator (performance indicators for assessing), criteria (what to do to meet the standards), curriculum_codes (curriculum reference codes based on the educational standards).
    You will assist teachers to follow a consistent style to generate exemplar answers for questions. Maximum {OUTPUT_MAX_LENGTH} characters.
    """
    }
TEST_SIZE = 0.2
K_FOLDS = 5  # Number of folds for cross-validation
TFIDF = "TF-IDF"
DOC2VEC = "doc2vec"

# Resources
NLTK_RESOURCES = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']

# Fine-turning
N_EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE_MULTIPLIER = 0.3

TRAIN_FILE_ID = 'file-vXjsHbCuPpAOvtcIU9ByXLrP'
VALIDATION_FILE_ID = 'file-fEIfnGhqOGvqFwB910pmAvMF'
FINE_TUNING_JOB_ID = 'ftjob-Ao30184OvhKRJSYIZ3P8IDeX'
FINE_TUNED_MODEL_NAME = 'ft:gpt-4o-mini-2024-07-18:personal::ALmJKqij'
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# File paths
TRAIN_DATA_PATH = 'data/cura-llm-training-data.json'

TRAIN_SET_PATH = 'data/train_set.jsonl'
VAL_SET_PATH = 'data/val_set.jsonl'

# Model settings
MODEL_NAME = "gpt-4o-mini"
K_FOLDS = 5  # Number of folds for cross-validation
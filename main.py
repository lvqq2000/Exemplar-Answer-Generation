import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.file_io import read_json, save_to_jsonl
from utils.data_processing import preprocess_data, clean_text
from utils.openai_api import OpenAIClient
from config import OPENAI_API_KEY, TRAIN_DATA_PATH, TRAIN_SET_PATH, VAL_SET_PATH, TRAIN_FILE_ID, VALIDATION_FILE_ID, FINE_TUNING_JOB_ID


def main():
    # Read input data
    openAI_client = OpenAIClient(api_key=OPENAI_API_KEY)

    #input_data = read_json(TRAIN_DATA_PATH)

    #preprocessed_data = preprocess_data(input_data)

    #train_data, validation_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

    #save_to_jsonl(train_data, TRAIN_SET_PATH)
    #save_to_jsonl(validation_data, VAL_SET_PATH)

    #openAI_client.upload_dataset(TRAIN_SET_PATH, VAL_SET_PATH)

    #openAI_client.do_fine_tuning(TRAIN_FILE_ID, VALIDATION_FILE_ID)

    #openAI_client.check_fine_tuning(FINE_TUNING_JOB_ID)

    #openAI_client.obtain_fine_tuned_model_name()

    while True:
        task_content = input("Enter the task content: ")
        question = input("Enter the question: ")
        rubric = input("Enter the rubric (in JSON format): ")
        answer = openAI_client.generate_answer(task_content, question, clean_text(rubric))
        if answer:
            print("Generated exemplar Answer:", answer)

if __name__ == "__main__":
    main()
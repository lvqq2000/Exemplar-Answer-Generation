import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.file_io import read_json, save_df_to_json
from utils.data_processing import preprocess_dataframe
from utils.prompt_generating import create_train_prompt
from utils.openai_api import OpenAIClient
from config import OPENAI_API_KEY, TRAIN_DATA_PATH, INPUT_DEFAULT_PATH, VAL_SET_PATH, OUTPUT_PATH

def main():
    # Read input data
    openAI_client = OpenAIClient(api_key=OPENAI_API_KEY)

    training_data = read_json(TRAIN_DATA_PATH)

    training_data_df = pd.DataFrame(training_data)

    train_df, validation_df = train_test_split(training_data_df, test_size=0.2, random_state=42)

    print("Saving validation data for testing...")
    save_df_to_json(validation_df, VAL_SET_PATH)
    save_df_to_json(validation_df, INPUT_DEFAULT_PATH)

    print("Preparing the training data...")
    preprocessed_train = preprocess_dataframe(train_df)

    train_prompt = create_train_prompt(preprocessed_train)

    # validation
    #openAI_client.generate_examplar_answers(validation_df, train_prompt)

    input_file_path = input(f"Enter the input file path (default: {INPUT_DEFAULT_PATH}):")
    if input_file_path:
        user_input = read_json(input_file_path)
    else:
        user_input = read_json(INPUT_DEFAULT_PATH)

    if user_input:
        input_df = pd.DataFrame(user_input)
        result_df = openAI_client.generate_examplar_answers(input_df, train_prompt)
        if result_df is not None:
            save_df_to_json(result_df, OUTPUT_PATH)

if __name__ == "__main__":
    main()
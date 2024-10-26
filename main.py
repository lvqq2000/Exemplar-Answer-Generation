import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.file_io import read_json, save_df_to_json
from utils.data_processing import preprocess_dataframe
from utils.prompt_generating import create_train_prompt
from utils.openai_api import OpenAIClient
from utils.evaluation import validation, k_fold_cross_validation, print_similarity_results
from config import *

def generate_exemplar_answers():
    print("Generating exemplar answers...")

    openAI_client = OpenAIClient(api_key=OPENAI_API_KEY)

    training_data = read_json(TRAIN_DATA_PATH)

    training_data_df = pd.DataFrame(training_data)

    train_df, validation_df = train_test_split(training_data_df, test_size=TEST_SIZE)

    print("Preparing the training data...")
    preprocessed_train = preprocess_dataframe(train_df)
    train_prompt = create_train_prompt(preprocessed_train)

    print("Saving validation data for testing...")
    save_df_to_json(validation_df, VAL_SET_PATH)
    save_df_to_json(validation_df, INPUT_DEFAULT_PATH)
    print("Validation data is ready for testing.")

    # validation
    #openAI_client.generate_examplar_answers(validation_df, train_prompt)

    input_file_path = input(f"Enter the input file path (default: {INPUT_DEFAULT_PATH}):") or INPUT_DEFAULT_PATH
    user_input = read_json(input_file_path)

    if user_input:
        input_df = pd.DataFrame(user_input)
        result_df = openAI_client.generate_examplar_answers(input_df, train_prompt=train_prompt)
        if result_df is not None:
            save_df_to_json(result_df, OUTPUT_PATH)
            # Ask the user if they want to perform validation
            do_validation = input(f"Do you want to perform validation on the results? (y/n): ")
            
            if do_validation.lower() in ['y', 'yes']:
                print("Performing validation on the results...")
                validation(training_data_df, result_df)

    return

def do_validation():
    # Implement the logic for validation
    print("Performing k fold cross validation on the LLM model...")

    training_data = read_json(TRAIN_DATA_PATH)

    training_data_df = pd.DataFrame(training_data)

    print("Without training:")
    client_without_training = OpenAIClient(api_key=OPENAI_API_KEY)
    result_without_training = k_fold_cross_validation(training_data_df, client_without_training, doTraining=False)

    client = OpenAIClient(api_key=OPENAI_API_KEY)
    result = k_fold_cross_validation(training_data_df, client, doTraining=True)

    print("With training:")
    print_similarity_results(result_without_training, doTraining=False)
    print_similarity_results(result, doTraining=True)

    print("\nNote: The folds for 'undergoes training' and 'does not undergo training' are randomly chosen and may not be the same.")
    return

def evaluate_generated_answers():
    file_path = input(f"Enter the path of the file you want to evaluate (default: {OUTPUT_PATH}):") or OUTPUT_PATH
    data = read_json(file_path)

    if data:
        data_df = pd.DataFrame(data)
        validation(data_df, data_df)

def main():
    # Ask the user what they want to do
    print("What would you like to do?")
    print("1: Generate exemplar answers")
    print("2: Do validation on the LLM model")
    print("3: Evaluate generated exemplar answers")
    
    choice = input("Please enter your choice (1, 2, or 3): ")

    if choice == '1':
        generate_exemplar_answers()
    elif choice == '2':
        do_validation()
    elif choice == '3':
        evaluate_generated_answers()

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
            

if __name__ == "__main__":
    main()
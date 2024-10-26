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
    """
    Generates exemplar answers using the OpenAI API based on user input.
    Optionally trains the model with provided training data, validates results, 
    and saves the generated answers.
    """
    print("Generating exemplar answers...")
    
    openAI_client = OpenAIClient(api_key=OPENAI_API_KEY)
    train_prompt = []

    # Ask user whether training should be performed
    do_training = input(f"Do you want to use the train data in {TRAIN_DATA_PATH} to train the LLM model? (y/n): ").strip().lower() in ['y', 'yes']
    
    if do_training:
        print("Preparing and validating training data...")
        
        # Load and preprocess training data
        training_data = read_json(TRAIN_DATA_PATH)
        training_data_df = pd.DataFrame(training_data)
        train_df, validation_df = train_test_split(training_data_df, test_size=TEST_SIZE)
        
        # Prepare training prompts and save validation data for testing
        preprocessed_train = preprocess_dataframe(train_df)
        train_prompt = create_train_prompt(preprocessed_train)
        
        save_df_to_json(validation_df, VAL_SET_PATH)
        print("Validation data saved and ready for testing.")
    
    # Load input data for generation
    input_file_path = input(f"Enter the input file path (default: {INPUT_DEFAULT_PATH}):") or INPUT_DEFAULT_PATH
    user_input = read_json(input_file_path)

    if user_input:
        input_df = pd.DataFrame(user_input)
        result_df = openAI_client.perform_generation(input_df, train_prompt=train_prompt)
        
        # Save results based on training condition
        if result_df is not None:
            save_df_to_json(result_df, OUTPUT_PATH if do_training else OUTPUT_PATH_WITHOUT_TRAINING)
            
            # Optionally validate generated answers
            if input("Do you want to perform validation on the results? (y/n): ").strip().lower() in ['y', 'yes']:
                print("Validating the generated results...")
                validation(result_df)

def do_validation():
    """
    Performs k-fold cross-validation on the language model to assess performance.
    Runs validation both with and without model training for comparison.
    """
    print(f"Performing {K_FOLDS}-fold cross-validation on the LLM model...")

    # Load training data
    training_data = read_json(TRAIN_DATA_PATH)
    training_data_df = pd.DataFrame(training_data)
    
    # Run cross-validation without training
    client_without_training = OpenAIClient(api_key=OPENAI_API_KEY)
    result_without_training = k_fold_cross_validation(training_data_df, client_without_training, do_training=False)

    # Run cross-validation with training
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    result = k_fold_cross_validation(training_data_df, client, do_training=True)
    
    # Display similarity results
    print_similarity_results(result_without_training, training_done=False)
    print_similarity_results(result, training_done=True)

    print("Note: Different folds may have been selected randomly for each training condition.")
    return

def evaluate_generated_answers():
    """
    Evaluates generated exemplar answers by comparing them with expected answers 
    in a given JSON file.
    """
    file_path = input(f"Enter the path of the file you want to evaluate (default: {OUTPUT_PATH}):") or OUTPUT_PATH
    data = read_json(file_path)

    if data:
        data_df = pd.DataFrame(data)
        validation(data_df)

def main():
    """
    Main menu function guiding the user to generate exemplar answers, 
    perform model validation, or evaluate generated answers.
    """
    print("What would you like to do?")
    print("1: Generate exemplar answers")
    print("2: Perform validation on the LLM model")
    print("3: Evaluate generated exemplar answers")
    
    choice = input("Please enter your choice (1, 2, or 3): ").strip()

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
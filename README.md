# Exemplar Answer Generation with OpenAI API

## Overview
This project aims to integrate the OpenAI API to generate exemplar answers for various questions. The program will automatically produce high-quality responses based on the provided input, which includes the context of student task content, the question, and the assessment rubrics, ensuring alignment with the rubrics used for evaluation.

## Contents
- [Structure](#structure)
- [Setup](#setup)
- [Usage Guideline](#usage-guideline)
- [Testing and Evaluation](#testing-and-evaluation)

## Structure
```
.
├── data                                # Stores input and training data files used in the project.
│   ├── input.json                      # Default file to for system to fetch input data.
│   └── training_data.json              # Training data for the OpenAI model.
├── output                              # Folder to store generated output files.
│   ├── example_output_no_training.json # Example output genereated by untrained model.
│   └── example_output.json             # Example output genereated by trained model.
├── llm                                 # Contains logic specific to large language model (LLM) interaction.
│   ├── __init__.py
│   ├── openai_api.py                   # Script to interact with the OpenAI API, managing API calls and responses.
│   └── prompt_generating.py            # Script for creating prompt messages to send to the OpenAI model.
├── utils                               # Helper functions and utilities to support various tasks across the project.
│   ├── __init__.py
│   ├── data_processing.py              # Contains functions to preprocess, clean, and format data before model input.
│   ├── evaluation.py                   # Script to evaluate model outputs by comparing predictions with expected results.
│   ├── file_io.py                      # Handles file input/output operations, like reading from and saving to JSON files.
│   └── question.py                     # Helper file for question-related processing and categorization.
├── .gitignore                          # Specifies files and directories to ignore in version control, like `.env` and `__pycache__`.
├── config.py                           # Stores configuration variables, constants, and paths used across the project.
├── main.py                             # Entry point of the application, orchestrating the workflow and executing main functions.
├── requirements.txt                    # Lists dependencies needed for the project.
└── README.md                           # Project documentation with setup instructions, usage guidelines, and overview of the project.
```

## Set up
### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.6 or higher**: You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: The package installer for Python (comes bundled with Python installation).

### Step 1: Clone the repository or download the files
Open your terminal or command prompt and run the following command to clone the repository:
```bash
git clone https://github.com/lvqq2000/Exemplar-Answer-Generation
cd Exemplar-Answer-Generation
```

### Step 2: Create a virtual environment (optional but recommended)
```bash
python -m venv venv
```

### Step 3: Active the virtual environment (if you have created a virtual environment)
**On Windows**
```bash
venv\Scripts\activate
```
**On macOS/Linux**
```bash
source venv/bin/activate
```

### Step 4: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 5: Set Up Environment Variables To Store OpenAI API key
Create a .env file in the project directory to store your OpenAI API key:
**On Windows**
```bash
echo OPENAI_API_KEY=your_api_key_here > .env
```
**On macOS/Linux**
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```
Replace `your_api_key_here` with your actual OpenAI API key.

### Step 6: Run the Application
With the setup complete, you can run the application using:
```bash
python main.py
```

### Step 7: Deactivate the Virtual Environment (if you have created and actived a virtual environment)
```bash
deactivate
```

## Usage Guideline
After executing, you will see a menu with options. Choose one of the following by entering the corresponding number:

1: Generate exemplar answers: This option will guide you through the process of generating exemplar answers using input data and optionally training the model with provided training data.

2: Perform validation on the LLM model: This option executes k-fold cross-validation on the language model to assess its performance. It compares results with and without model training.

3: Evaluate generated exemplar answers: This option allows you to evaluate the answers generated by the model by comparing them with expected answers from a specified file.

### Generating Exemplar Answers
To generate exemplar answers using the OpenAI API, follow these steps:

1. **Start the Process**: Select the first option from the main menu by entering `1`.
2. **Training Data Usage**: You will be prompted to decide whether to use the training data located at `TRAIN_DATA_PATH` for training the model. Enter `y` to proceed with training or `n` to skip this step.
3. **Input File Path**: Provide the path for the input file. If you want to use the default path, simply press `Enter`.
4. **Results Generation**: After processing, the generated answers will be saved in the specified output location. If you opted for training, the validation data will also be saved for future testing.
5. **Optional Validation**: You will have the option to validate the generated results. If you wish to perform validation, enter `y`; otherwise, enter `n`.

### Performing Model Validation
To validate the model's performance, do the following:

1. **Initiate Validation**: Select the second option from the main menu by entering `2`.
2. **Cross-Validation Process**: The script will execute k-fold cross-validation on the language model using the provided training data. This validation will be conducted in two scenarios: with model training and without.
3. **View Results**: After the validation, similarity results from both scenarios will be displayed, allowing for a comparison of the model's performance.

### Evaluating Generated Answers
To evaluate the quality of generated answers, proceed with these steps:

1. **Choose Evaluation**: Select the third option from the main menu by entering `3`.
2. **Input File Path**: You will be prompted to enter the path of the file that contains the generated answers. To use the default output path, simply press `Enter`.
3. **Comparison**: The script will compare the generated answers with the expected answers found in the specified JSON file, providing an assessment of the model's output quality.


## Testing and Evaluation
The model uses k-fold cross-validation (default: 5) to assess performance. This validation will be conducted in two scenarios: with model training and without. For each fold, it will use the Doc2Vec model and TF-IDF model to measure the similarity between the generated answers and the sample training data. The mean similarity score will be calculated and printed to the concole once all folds are processed.

To perform the evaluation, see [Performing Model Validation](#performing-model-validation).

### Sample outputs for evaluation
Sample outputs produced by both the untrained model and the trained model have been provided as [example_output_no_training.json](output/example_output_no_training.json) and [example_output.json](output/example_output.json). Both files contain 20 identical questions, along with sample answers and generated answers from the model.
import os
from openai import OpenAI

from utils.prompt_generating import create_user_query
from utils.data_processing import preprocess_dataframe, show_progress_bar, clean_text, remove_long_entries
from config import MODEL_NAME, N_EPOCHS, BATCH_SIZE, LEARNING_RATE_MULTIPLIER, OUTPUT_ROW_NAME, TASK_CONTENT_MAX_LENGTH

class OpenAIClient:
    def __init__(self, api_key):
        """
        Initializes the OpenAIClient instance with an API key.

        Args:
            api_key (str): The API key for authenticating with the OpenAI API.
        """
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, messages):
        """Generates an answer based on the input messages by calling the OpenAI API."""
        
        try:
            # Request completion from the model
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            # Log the error if generation fails
            print(f"Error in generate_answer: {e}")
            return None

    def perform_generation(self, tasks_df, train_prompt=[], do_print=True):
        """Generates exemplar answers for tasks based on prompts and task contents."""

        if do_print:
            print("Preprocessing...")
        
        # Preprocess and clean task data
        preprocessed_df = preprocess_dataframe(tasks_df.copy())
        preprocessed_df['task_content'] = preprocessed_df['task_content'].apply(clean_text)
        preprocessed_df = remove_long_entries(preprocessed_df, 'task_content', length_limit=TASK_CONTENT_MAX_LENGTH, do_print=do_print)

        first_row = True
        grouped_tasks = preprocessed_df.groupby('task_id')
        generated_answers_dict = {}

        if do_print:
            print("Generating exemplar answers...")

        iteration = 0
        for task_id, group in grouped_tasks:
            # Generate answers for each question within the grouped tasks
            for index, row in group.iterrows():
                iteration += 1
                show_progress_bar(iteration, preprocessed_df.shape[0])

                # Create query messages based on task content and context
                if first_row:
                    first_row = False
                    generated_answer = self.generate_answer(train_prompt + [create_user_query(row, task_content=row['task_content'])])
                elif index == 0:
                    generated_answer = self.generate_answer([create_user_query(row, task_content=row['task_content'])])
                else:
                    generated_answer = self.generate_answer([create_user_query(row)])

                if not generated_answer:
                    return None
                # Store generated answer by question_id
                generated_answers_dict[row['question_id']] = generated_answer
        
        # Assign generated answers to new DataFrame column
        new_tasks_df = tasks_df.assign(**{OUTPUT_ROW_NAME: tasks_df['question_id'].map(generated_answers_dict)})
        
        # Remove entries with missing answers
        new_tasks_df = new_tasks_df.dropna(subset=[OUTPUT_ROW_NAME])

        return new_tasks_df

    def upload_dataset(self, train_file_path, validation_file_path):
        """Uploads training and validation files to the OpenAI API for fine-tuning."""

        try:
            # Upload training file
            self.train_file = self.client.files.create(
                file=open(train_file_path, "rb"),
                purpose="fine-tune"
            )
            
            # Upload validation file
            self.valid_file = self.client.files.create(
                file=open(validation_file_path, "rb"),
                purpose="fine-tune"
            )

            print(f"Training file Info: {self.train_file}")
            print(f"Validation file Info: {self.valid_file}")

        except Exception as e:
            print(f"Error in upload_dataset: {e}")

    def do_fine_tuning(self, train_file_id, validation_file_id):
        """Initiates a fine-tuning job with the specified training and validation file IDs."""
        try:
            # Create fine-tuning job with specified parameters
            model = self.client.fine_tuning.jobs.create(
                training_file=train_file_id,
                validation_file=validation_file_id,
                model=MODEL_NAME,
                hyperparameters={
                    "n_epochs": N_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate_multiplier": LEARNING_RATE_MULTIPLIER
                }
            )
            job_id = model.id
            status = model.status

            print(f'Fine-tuning model with jobID: {job_id}.')
            print(f"Training Response: {model}")
            print(f"Training Status: {status}")

        except Exception as e:
            print(f"Error in fine_tuning: {e}")
            return None

    def check_fine_tuning(self, job_id):
        """Retrieves and displays the status of a specific fine-tuning job."""

        # Retrieve fine-tuning job details
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        
        # Display job information
        job_id = job.id
        status = job.status
        model = job.model
        created_at = job.created_at

        print(f"Fine-tuning Job ID: {job_id}")
        print(f"Model: {model}")
        print(f"Status: {status}")
        print(f"Created At: {created_at}")

    def obtain_fine_tuned_model_name(self):
        """Lists all fine-tuning jobs and prints out job information to identify fine-tuned models."""

        result = self.client.fine_tuning.jobs.list()
        print(f"Fine_tuning.jobs: {result.data}")

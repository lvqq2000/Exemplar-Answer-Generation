import os
from openai import OpenAI

from utils.prompt_generating import create_user_query
from utils.data_processing import preprocess_dataframe, show_progress_bar, clean_text, remove_long_entries
from config import MODEL_NAME, N_EPOCHS, BATCH_SIZE, LEARNING_RATE_MULTIPLIER, OUTPUT_ROW_NAME, TASK_CONTENT_MAX_LENGTH

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            return None
        
    def perfrom_generation(self, tasks_df, train_prompt=[], do_print=True):
        if do_print:
            print("Preprocessing...")
        preprocessed_df = preprocess_dataframe(tasks_df.copy())

        # Remove any questions that have a 'task_content' which exceed a specified length limit
        preprocessed_df['task_content'] = preprocessed_df['task_content'].apply(clean_text)
        preprocessed_df = remove_long_entries(preprocessed_df, 'task_content', length_limit=TASK_CONTENT_MAX_LENGTH, do_print=do_print)

        first_row = True

        grouped_tasks = preprocessed_df.groupby('task_id')

        generated_answers_dict = {}

        if do_print:
            print("Generating examplar answers...")

        iteration = 0
        for task_id, group in grouped_tasks:
            for index, row in group.iterrows():
                iteration += 1

                show_progress_bar(iteration, preprocessed_df.shape[0])

                if first_row:
                    first_row = False
                    generated_answer = self.generate_answer(train_prompt + [create_user_query(row, task_content=row['task_content'])])
                elif index == 0:
                    generated_answer = self.generate_answer([create_user_query(row, task_content=row['task_content'])])
                else:
                    generated_answer = self.generate_answer([create_user_query(row)])

                if not generated_answer:
                    return None
                generated_answers_dict[row['question_id']] = generated_answer
        
        new_tasks_df = tasks_df.assign(**{OUTPUT_ROW_NAME: tasks_df['question_id'].map(generated_answers_dict)})

        # Drop rows where there is no generated exemplar_answer
        new_tasks_df = new_tasks_df.dropna(subset=[OUTPUT_ROW_NAME])

        return new_tasks_df

    def upload_dataset(self, train_file_path, validation_file_path):
        try:
            self.train_file = self.client.files.create(
                file=open(train_file_path, "rb"),
                purpose="fine-tune"
            )

            self.valid_file = self.client.files.create(
                file=open(validation_file_path, "rb"),
                purpose="fine-tune"
            )

            print(f"Training file Info: {self.train_file}")
            print(f"Validation file Info: {self.valid_file}")

        except Exception as e:
            print(f"Error in upload_dataset: {e}")

    def do_fine_tuning(self, train_file_id, validation_file_id):
        try:
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
        # Retrieve the fine-tuning job details
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        
        # Extract relevant information
        job_id = job.id
        status = job.status
        model = job.model
        created_at = job.created_at

        print(f"Fine-tuning Job ID: {job_id}")
        print(f"Model: {model}")
        print(f"Status: {status}")
        print(f"Created At: {created_at}")

    def obtain_fine_tuned_model_name(self):
        result = self.client.fine_tuning.jobs.list()
        print(f"Fine_tuning.jobs: {result.data}")
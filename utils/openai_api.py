import os
from openai import OpenAI

from config import MODEL_NAME, N_EPOCHS, BATCH_SIZE, LEARNING_RATE_MULTIPLIER, FINE_TUNED_MODEL_NAME

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, task_content, question, rubric):
        try:
            completion = self.client.chat.completions.create(
                model=FINE_TUNED_MODEL_NAME,
                messages=[{"role": "system", "content": "Generate an exemplar answer for the question based on the given task content, ensuring it aligns with the rubric. Rubric include items (Key performance indicators for assessing the response. items[0] is for the highest score), criteria (Describes what to do to meet the assessment standards), curriculum_codes (Curriculum reference codes based on the educational standards, which might differ by region)"},
                    {"role": "user", "content": f"\"task content: {task_content}; question: {question}; rubric: {rubric}\""}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            return None

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
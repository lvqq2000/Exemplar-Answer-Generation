import pandas as pd

from utils.question import QuestionType
from config import QUESTION_CATEGORY_NAME

def create_train_prompt(df):
    """
    Creates a training prompt based on sample questions from each question type in the dataset.
    The prompt provides the model with a selection of questions, expected answers, and rubrics 
    for generating exemplar responses aligned with educational standards.
    """
    print("Creating prompts based on training data...")

    question_list = pd.DataFrame()

    # Select random questions by type for training prompt, including all for 'Other'
    for question_type in QuestionType:
        if question_type != QuestionType.OTHER:
            filtered_questions = df[df[QUESTION_CATEGORY_NAME] == question_type]
            if not filtered_questions.empty:
                n_sample = min(len(filtered_questions), 2)
                random_questions = filtered_questions.sample(n=n_sample)
                question_list = pd.concat([question_list, random_questions], ignore_index=True)
        else:
            other_questions = df[df[QUESTION_CATEGORY_NAME] == QuestionType.OTHER]
            question_list = pd.concat([question_list, other_questions], ignore_index=True)

    # Construct prompt with question examples and rubric details
    prompt = """
    You will assist teachers in generating exemplar answers for questions.
    The answer should align with the rubric, which includes performance indicators (assessment criteria), 
    criteria (standards to meet), and curriculum reference codes (educational standards).
    
    The following are various types of questions, along with expected answers and rubrics. 
    Learn the styles (structure and length) used for different types of questions and apply them to new tasks.
    """

    # Include each sampled question and rubric details in the prompt
    for index, row in question_list.iterrows():
        prompt += f"""
        {index + 1}.
        User input:
        Question: {row['question']}

        Rubrics:
            Performance indicator: {row['indicator']}
            Criteria: {row['criteria']}
            Curriculum Reference Codes: {row['curriculum_codes']}

        Expected output (answer):
        {row['answer']}
        """

    prompt += """
    You will also be provided with task content, which you should use to generate answers.
    """
    
    return [{
        "role": "system", 
        "content": prompt
    }]

def create_user_query(row, task_content=None):
    """
    Constructs a query for generating an exemplar answer based on a specific question, rubric, 
    and task content if provided. Uses 'same as above' if task content is unavailable.
    """
    task_content_message = f"Task content: {task_content}" if task_content else "Task content: same as above"
    
    return {
        "role": "user",
        "content": f"""
                    {task_content_message}

                    Question: {row['question']}

                    Rubric:
                        Performance indicator: {row['indicator']}
                        Criteria: {row['criteria']}
                        Curriculum_codes: {row['curriculum_codes']}

                    Generate a high-quality exemplar answer (plain text without formatting) for the question 
                    using the given task content, ensuring it aligns with the rubric.
                    """
    }
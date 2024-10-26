import pandas as pd

from utils.question import QuestionType
from config import QUESTION_CATEGORY_NAME

def create_train_prompt(df):
    print("Creating prompts based on training data...")

    # Initialize a DataFrame to hold the selected random questions
    question_list = pd.DataFrame()

    # Select a random question for each type, or all for 'Other'
    for question_type in QuestionType:
        if question_type != QuestionType.OTHER:
            # Filter questions for the current type
            filtered_questions = df[df[QUESTION_CATEGORY_NAME] == question_type]
            
            if not filtered_questions.empty:
                # Select maximum 2 random questions for the type
                n_sample = min(len(filtered_questions), 2)
                random_questions = filtered_questions.sample(n=n_sample)
                question_list = pd.concat([question_list, random_questions], ignore_index=True)
        else:
            # Select all 'Other' questions
            other_questions = df[df[QUESTION_CATEGORY_NAME] == QuestionType.OTHER]
            question_list = pd.concat([question_list, other_questions], ignore_index=True)


    # Prepare the prompt with the selected random questions
    prompt = f"""
    You will assist teachers to generate exemplar answers for questions.
    The answer need to align with the rubric, which includes indicator (performance indicators for assessing), criteria (what to do to meet the standards), curriculum_codes (curriculum reference codes based on the educational standards).
    
    Learn from the following examples of questions, and their corresponding answers:

    """

    # Loop through the random_questions DataFrame to include each question
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

    prompt += f"""
    Please understand this style and apply it to new tasks.

    You will be also provided with the task content, which you will be using to answer the questions.
    """

    return [{
        "role": "system", 
        "content": prompt
        }]

def create_user_query(row, task_content=None):
    if task_content:
        task_content_message = "Task content: " + task_content
    else:
        task_content_message = "Task content: same as above"
    return {
                "role": "user",
                "content": 
                    f"""
                    {task_content_message}

                    Question: {row['question']}

                    Rubric:
                        Performance indicator: {row['indicator']}
                        Criteria: {row['criteria']}
                        Curriculum_codes: {row['curriculum_codes']}

                    Generate a high-quality exemplar answer (plain text without formatting) for the question using the given task content, ensuring it aligns with the rubric.
                    """
            }
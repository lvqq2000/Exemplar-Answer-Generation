from config import SYSTEM_PROMPT

def create_train_prompt(df):
    print("Creating prompts based on training data...")

    grouped_tasks = df.groupby('task_content')
    train_prompt = [SYSTEM_PROMPT]
    for task_content, group in grouped_tasks:
        for index, row in group.iterrows(): 
            if index == 0:
                train_prompt.append(create_user_query(row, task_content=task_content))
            else:
                train_prompt.append(create_user_query(row))

            train_prompt.append({
                "role": "assistant",
                "content": row['answer']
            })
    return train_prompt

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
                    Indicator: {row['indicator']}
                    Criteria: {row['criteria']}
                    Curriculum_codes: {row['curriculum_codes']}

                    Generate a succint exemplar answer (plain text without formatting) for the question using only the given task content, ensuring it aligns with the rubric.
                    """
            }
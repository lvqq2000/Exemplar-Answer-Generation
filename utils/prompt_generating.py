from config import ANSWER_MAX_LENGTH

def create_train_prompt(df):
    print("Creating prompts based on training data...")

    grouped_tasks = df.groupby('task_content')
    train_prompt = [{"role": "system", "content": f"You will be provided with a task content, a question related to task content and the rubric which includes indicator (performance indicators for assessing), criteria (what to do to meet the standards), curriculum_codes (curriculum reference codes based on the educational standards). You will assist teachers to follow a consistent style to generate exemplar answers for questions. Maximum {ANSWER_MAX_LENGTH} characters. "}]
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
    print("Done")
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
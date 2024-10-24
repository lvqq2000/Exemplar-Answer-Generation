import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_to_jsonl(data, output_file_path):
    jsonl_data = []
    for index, row in data.iterrows():
        jsonl_data.append({
            "messages": [
                {"role": "system", "content": "Generate an exemplar answer for the question based on the given task content, ensuring it aligns with the rubric. Rubric include items (Key performance indicators for assessing the response. items[0] is for the highest score), criteria (Describes what to do to meet the assessment standards), curriculum_codes (Curriculum reference codes based on the educational standards, which might differ by region)"},
                {"role": "user", "content": f"\"task content: {row['task_content']}; question: {row['question']}; rubric: {row['rubric']}\""},
                {"role": "assistant", "content": f"\"{row['answer']}\""}

            ]
        })

    # Save to JSONL format
    with open(output_file_path, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
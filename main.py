import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.file_io import read_json
from utils.openai_api import generate_answer
from utils.data_processing import preprocess_data
from config import TRAIN_DATA_PATH


def main():
    # Read input data
    input_data = read_json(TRAIN_DATA_PATH)

    preprocessed_data = preprocess_data(input_data)

    while False:
        question = input("Enter your question: ")
        answer = generate_answer(question)
        if answer:
            print("Generated exemplar Answer:", answer)

if __name__ == "__main__":
    main()
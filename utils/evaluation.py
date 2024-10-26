import numpy as np
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import K_FOLDS, OUTPUT_ROW_NAME, EXPECTED_ROW_NAME, TFIDF, DOC2VEC
from utils.data_processing import preprocess_dataframe, preprocess_text
from utils.prompt_generating import create_train_prompt

def train_doc2vec_model(data):
    """Trains a Doc2Vec model using preprocessed text from the input DataFrame."""
    documents = [
        TaggedDocument(words=preprocess_text(row[EXPECTED_ROW_NAME]), tags=[str(row["question_id"])])
        for _, row in data.iterrows()
    ]
    
    # Create and train the Doc2Vec model
    model = Doc2Vec(vector_size=50, window=2, min_count=1, workers=4, epochs=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model

# Function to evaluate generated answers
def evaluate_doc2vec_similarity(model, generated_answers, expected_answers):
    """Evaluates generated answers against expected answers using the trained Doc2Vec model.
    
    Args:
        model: The trained Doc2Vec model.
        generated_answers (list of str): List of generated answer texts.
        expected_answers (list of str): List of expected answer texts.

    Returns:
        list: Similarity scores between generated and expected answers.
    """

    preprocessed_generated = [preprocess_text(answer) for answer in generated_answers]
    preprocessed_expected = [preprocess_text(answer) for answer in expected_answers]

    results = []
    
    for generated, expected in zip(preprocessed_generated, preprocessed_expected):
        # Calculate similarity using Doc2Vec
        generated_vector = model.infer_vector(generated)
        expected_vector = model.infer_vector(expected)
        
        similarity = 1 - cosine(generated_vector, expected_vector)
        results.append(similarity)
    
    return results

def evaluate_tfidf_similarity(generated_answers, expected_answers):
    preprocessed_generated = [preprocess_text(answer) for answer in generated_answers]
    preprocessed_expected = [preprocess_text(answer) for answer in expected_answers]

    similarity_scores = []

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Loop through each pair of generated and expected answers
    for generated, expected in zip(preprocessed_generated, preprocessed_expected):
        generated_texts = ' '.join(generated)
        expected_texts = ' '.join(expected)

        tfidf_matrix = tfidf_vectorizer.fit_transform([generated_texts, expected_texts])

        # Calculate cosine similarity for the current pair
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        similarity_scores.append(similarity)

    return similarity_scores

def validation(doc2vec_model_data, result_df):
    doc2vec_model = train_doc2vec_model(doc2vec_model_data)

    try:
        # Check if the DataFrame is not empty
        if result_df.empty:
            raise ValueError("The DataFrame is empty.")

        # Extract generated and expected lists
        generated = result_df[OUTPUT_ROW_NAME].tolist()
        expected = result_df[EXPECTED_ROW_NAME].tolist()

    except KeyError as e:
        print(f"Error: The specified column '{e.args[0]}' does not exist in the DataFrame.")
        return None
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Proceed with evaluation if no exceptions were raised
    doc2vec_evaluation_results = evaluate_doc2vec_similarity(doc2vec_model, generated, expected)
    print("Mean Doc2vec similarity score is ", np.mean(doc2vec_evaluation_results))

    tfidf_evaluation_results = evaluate_tfidf_similarity(generated, expected)
    print("TF_IDF similarity score is ", np.mean(tfidf_evaluation_results))


def k_fold_cross_validation(data, openAI_client, n_splits=K_FOLDS, do_training=True):
    kf = KFold(n_splits=n_splits)

    doc2vec_model = train_doc2vec_model(data)

    results = {DOC2VEC: [], TFIDF: []}
    for fold_index, (train_index, val_index) in enumerate(kf.split(data), start=1):
        print(f"Processing Fold {fold_index}/{n_splits}...")

        train_data, val_data = data.iloc[train_index], data.iloc[val_index]

        if do_training:
            preprocessed_train = preprocess_dataframe(train_data)
            train_prompt = create_train_prompt(preprocessed_train)

            result_df = openAI_client.generate_examplar_answers(val_data, train_prompt=train_prompt, do_print=False)
        else:
            result_df = openAI_client.generate_examplar_answers(val_data, do_print=False)

        generated = result_df[OUTPUT_ROW_NAME].tolist()
        expected = result_df[EXPECTED_ROW_NAME].tolist()

        doc2vec_evaluation_results = evaluate_doc2vec_similarity(doc2vec_model, generated, expected)
        results[DOC2VEC].append(np.mean(doc2vec_evaluation_results))

        tfidf_evaluation_results = evaluate_tfidf_similarity(generated, expected)
        results[TFIDF].append(np.mean(tfidf_evaluation_results))

    return results

def print_similarity_results(results, training_done=True):
    if training_done:
        print("When the LLM model undergoes training:")
    else:
        print("When the LLM model doesn't undergo training:")


    for method, result in results.items():
        mean_similarity = sum(result) / len(result) if results else 0
        print(f"{method} mean similarity across {K_FOLDS} folds: {mean_similarity:.5f}")
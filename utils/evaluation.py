import numpy as np
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import K_FOLDS, OUTPUT_ROW_NAME, EXPECTED_ROW_NAME, TFIDF, DOC2VEC
from utils.data_processing import preprocess_dataframe, preprocess_text
from llm.prompt_generating import create_train_prompt

def train_doc2vec_model(data):
    """Trains a Doc2Vec model using preprocessed text from the input DataFrame."""

    # Group data by 'task_id' and create TaggedDocument for each group
    grouped_tasks = data.groupby('task_id')
    documents = [
        TaggedDocument(words=preprocess_text(group['task_content'].iloc[0]), tags=[str(task_id)])
        for task_id, group in grouped_tasks
    ]

    # Initialize and train Doc2Vec model
    model = Doc2Vec(vector_size=30, window=2, min_count=1, workers=4, epochs=20)
    model.build_vocab(documents)  # Build vocabulary based on the documents
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)  # Train model

    return model

def evaluate_doc2vec_similarity(model, generated_answers, expected_answers):
    """Evaluates generated answers against expected answers using the trained Doc2Vec model."""

    # Preprocess both generated and expected answers
    preprocessed_generated = [preprocess_text(answer) for answer in generated_answers]
    preprocessed_expected = [preprocess_text(answer) for answer in expected_answers]

    results = []
    
    # Calculate similarity for each pair of answers
    for generated, expected in zip(preprocessed_generated, preprocessed_expected):
        # Infer vectors for generated and expected answers
        generated_vector = model.infer_vector(generated)
        expected_vector = model.infer_vector(expected)
        
        # Calculate cosine similarity and normalize it to a score
        cosine_distance = cosine(generated_vector, expected_vector)
        similarity = 1 - (cosine_distance / 2)  # Normalize to a [0, 1] range

        results.append(similarity)
    
    return results

def evaluate_tfidf_similarity(generated_answers, expected_answers):
    """Evaluates similarity between generated and expected answers using TF-IDF and cosine similarity."""

    # Preprocess both generated and expected answers
    preprocessed_generated = [preprocess_text(answer) for answer in generated_answers]
    preprocessed_expected = [preprocess_text(answer) for answer in expected_answers]

    similarity_scores = []

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Calculate similarity for each pair of answers
    for generated, expected in zip(preprocessed_generated, preprocessed_expected):
        # Join tokens back into a single string for TF-IDF
        generated_texts = ' '.join(generated)
        expected_texts = ' '.join(expected)

        # Create TF-IDF matrix for the two texts
        tfidf_matrix = tfidf_vectorizer.fit_transform([generated_texts, expected_texts])

        # Compute cosine similarity between generated and expected TF-IDF vectors
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        similarity_scores.append(similarity)

    return similarity_scores

def validation(result_df):
    """Validates a dataset by calculating Doc2Vec and TF-IDF similarity scores."""

    # Train a Doc2Vec model on the result DataFrame
    doc2vec_model = train_doc2vec_model(result_df)

    try:
        # Check if DataFrame is empty
        if result_df.empty:
            raise ValueError("The DataFrame is empty.")

        # Extract generated and expected answers from DataFrame
        generated = result_df[OUTPUT_ROW_NAME].tolist()
        expected = result_df[EXPECTED_ROW_NAME].tolist()

    # Handle exceptions for missing columns or empty data
    except KeyError as e:
        print(f"Error: The specified column '{e.args[0]}' does not exist in the DataFrame.")
        return None
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Evaluate Doc2Vec similarity and print the mean score
    doc2vec_evaluation_results = evaluate_doc2vec_similarity(doc2vec_model, generated, expected)
    print("Mean Doc2Vec similarity score is ", np.mean(doc2vec_evaluation_results))

    # Evaluate TF-IDF similarity and print the mean score
    tfidf_evaluation_results = evaluate_tfidf_similarity(generated, expected)
    print("Mean TF-IDF similarity score is ", np.mean(tfidf_evaluation_results))

def k_fold_cross_validation(data, openAI_client, n_splits=K_FOLDS, do_training=True):
    """Performs k-fold cross-validation to evaluate model performance across multiple splits."""

    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits)

    # Train a Doc2Vec model on the full dataset for evaluation
    doc2vec_model = train_doc2vec_model(data)

    # Initialize dictionary to store results for each method
    results = {DOC2VEC: [], TFIDF: []}
    for fold_index, (train_index, val_index) in enumerate(kf.split(data), start=1):
        print(f"Processing Fold {fold_index}/{n_splits}...")

        # Split data into training and validation sets
        train_data, val_data = data.iloc[train_index], data.iloc[val_index]

        # If training, generate results for validation data with OpenAI client
        if do_training:
            preprocessed_train = preprocess_dataframe(train_data, do_print=False)
            train_prompt = create_train_prompt(preprocessed_train)

            result_df = openAI_client.perform_generation(val_data, train_prompt=train_prompt, do_print=False)
        else:
            result_df = openAI_client.perform_generation(val_data, do_print=False)

        # Extract generated and expected answers for evaluation
        generated = result_df[OUTPUT_ROW_NAME].tolist()
        expected = result_df[EXPECTED_ROW_NAME].tolist()

        # Calculate and store Doc2Vec similarity score
        doc2vec_evaluation_results = evaluate_doc2vec_similarity(doc2vec_model, generated, expected)
        results[DOC2VEC].append(np.mean(doc2vec_evaluation_results))

        # Calculate and store TF-IDF similarity score
        tfidf_evaluation_results = evaluate_tfidf_similarity(generated, expected)
        results[TFIDF].append(np.mean(tfidf_evaluation_results))

    return results

def print_similarity_results(results, training_done=True):
    """Prints the mean similarity score for each method across k-folds."""

    if training_done:
        print("\nWhen the LLM model undergoes training:")
    else:
        print("\nWhen the LLM model doesn't undergo training:")

    # Calculate and print mean similarity score for each method
    for method, result in results.items():
        mean_similarity = sum(result) / len(result) if results else 0
        print(f"{method} mean similarity across {K_FOLDS} folds: {mean_similarity:.5f}")

    print()  # New line for readability
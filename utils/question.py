from enum import Enum

class QuestionType(Enum):
    # Enum representing different types of questions for categorization.
    YES_NO = "Yes or No"
    ANALYSIS = "Analysis"
    EXPLANATION = "Explanation"
    EXPLORATORY = "Exploratory"
    IDENTIFICATION = "Identification"
    PRACTICAL = "Practical"
    COMPARISON = "Comparison"
    CREATIVE = "Creative"
    CLASSIFICATION = "Classification"
    OTHER = "Other"

KEYWORDS = {
    QuestionType.YES_NO: ["do", "does", "will"],
    QuestionType.ANALYSIS: ["calculate", "determine", "measure", "compare"],
    QuestionType.EXPLANATION: ["explain", "describe", "summarize", "draw", "diagram", "illustrate", "reason"],
    QuestionType.EXPLORATORY: ["why", "how", "think", "hypothesis", "predict", "speculate"],
    QuestionType.IDENTIFICATION: ["identify", "name", "label", "which"],
    QuestionType.PRACTICAL: ["devise", "hypothesis", "experiment", "design", "apply"],
    QuestionType.COMPARISON: ["compare", "contrast", "similarities", "differences", "relative"],
    QuestionType.CREATIVE: ["imagine", "innovate"],
    QuestionType.CLASSIFICATION: ["classify", "group", "categorize"]
}

def categorize_question(question):
    """Categorizes a question based on keywords."""
    question = question.lower()
    for question_type, keywords in KEYWORDS.items():
        if any(keyword in question for keyword in keywords):
            return question_type
    return QuestionType.OTHER
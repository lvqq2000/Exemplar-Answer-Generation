# Exemplar Answer Generation with OpenAI API

## Overview
This project aims to integrate the OpenAI API to generate exemplar answers for various questions. The program will automatically produce high-quality responses based on the provided input, which includes the context of student task content, the question, and the assessment rubrics, ensuring alignment with the rubrics used for evaluation.

## Contents
- [Structure](#structure)
- [Setup](#setup)
- [Testing and Evaluation](#testing-and-evaluation)

## Set up
### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.6 or higher**: You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: The package installer for Python (comes bundled with Python installation).

### Step 1: Clone the repository or download the files
Open your terminal or command prompt and run the following command to clone the repository:
```bash
git clone https://github.com/lvqq2000/Exemplar-Answer-Generation
cd Exemplar-Answer-Generation
```

### Step 2: Create a virtual environment (optional but recommended)
```bash
python -m venv venv
```

### Step 3: Active the virtual environment (if you have created a virtual environment)
**On Windows**
```bash
venv\Scripts\activate
```
**On macOS/Linux**
```bash
source venv/bin/activate
```

### Step 4: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 5: Set Up Environment Variables To Store OpenAI API key
Create a .env file in the project directory to store your OpenAI API key:
**On Windows**
```bash
echo OPENAI_API_KEY=your_api_key_here > .env
```
**On macOS/Linux**
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```
Replace `your_api_key_here` with your actual OpenAI API key.

### Step 6: Run the Application
With the setup complete, you can run the application using:
```bash
python main.py
```

### Step 7: Deactivate the Virtual Environment (if you have created and actived a virtual environment)
```bash
deactivate
```

## Testing and Evaluation

Deployment Instructions – Researchly Research Abstract Classifier

1. Application Overview

The Researchly Research Abstract Classifier is a web application developed using Streamlit. It classifies research abstracts into one of two categories:

Artificial Intelligence: Research related to artificial intelligence and its applications.

Classical Research: Traditional research that does not primarily focus on artificial intelligence.

The application uses a Support Vector Machine (SVM) classifier with Term Frequency–Inverse Document Frequency (TF-IDF) feature extraction to classify user-submitted research abstracts.

2. Project Structure

The application requires the following project structure:

ITNPA4-Project-1/
│
├── Datasets/
│   └── research_abstracts.json
│
└── Deployment/
    ├── classifier.py
    ├── requirements.txt
    └── README.md

The dataset must remain inside the Datasets folder for the application to locate it correctly.

3. Requirements

The following software and Python libraries are required:

Python 3.x

Streamlit

Pandas

NumPy

NLTK

Scikit-learn

4. Installation Instructions

Step 1: Download the repository

Clone the GitHub repository using:

git clone YOUR_GITHUB_REPOSITORY_URL

Alternatively, download the repository as a ZIP file from GitHub and extract it.

Step 2: Open the project directory

Open a terminal and navigate to the main project directory:

cd ITNPA4-Project-1

Step 3: Install dependencies

Install the required Python libraries using:

pip install -r Deployment/requirements.txt

Step 4: Download NLTK resources

Download the required NLTK resources by executing:

python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

5. Running the Application

Ensure that the terminal is currently inside the main ITNPA4-Project-1 directory.

Execute the following command:

streamlit run Deployment/classifier.py

Streamlit will start the application and provide a local URL, usually:

http://localhost:8501

Open this URL in a web browser to access the Researchly Research Abstract Classifier.

6. Using the Application

Open the application in your web browser.

Enter a research abstract into the text input area.

Click the Classify Abstract button.

The application will process the abstract and display its predicted category.

The predicted category will be either Artificial Intelligence or Classical Research.

7. Technologies Used

Python

Streamlit

Pandas

NumPy

NLTK

Scikit-learn

TF-IDF

Linear Support Vector Classification (LinearSVC)
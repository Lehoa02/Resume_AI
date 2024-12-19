# README for Resume Screening Project

## Overview
This project is an AI-powered Resume Screening Tool that automates the evaluation of resumes. It processes resumes in PDF or TXT format, extracts structured information, categorizes resumes into predefined industries, recommends suitable job roles, and identifies key details such as skills and education. The tool leverages Flask for the web interface and Random Forest classifiers for machine learning tasks.

---

## Features
1. **Resume Categorization**: Classifies resumes into predefined categories such as IT, Engineering, and Marketing.
2. **Job Recommendation**: Suggests relevant job roles based on extracted skills.
3. **Information Extraction**: Extracts contact information, skills, and education details using regex and predefined keyword lists.
4. **Skill Matching**: Compares skills from resumes to job descriptions, providing match scores.

---

## Dependencies
The project requires the following:
- Python 3.8+
- Flask
- Scikit-learn
- PyPDF2
- regex
- docx2txt
- pickle (for loading pre-trained models)

---

## Setup
1. **Install Required Libraries**:
   ```bash
   pip install flask scikit-learn PyPDF2 regex docx2txt
   ```

2. **Folder Structure**:
   Ensure the following folder structure:
   ```
   project-root/
   |-- app.py
   |-- templates/
   |   |-- resume.html
   |-- uploads/
   |-- models/
       |-- rf_classifier_categorization.pkl
       |-- tfidf_vectorizer_categorization.pkl
       |-- rf_classifier_job_recommendation.pkl
       |-- tfidf_vectorizer_job_recommendation.pkl
   ```

3. **Place Models**:
   Download and place the trained models in the `models/` directory.

---

## Running the Application
1. **Start the Flask Server**:
   ```bash
   python app.py
   ```

2. **Access the Application**:
   Open a web browser and go to `http://127.0.0.1:5000/`.

---

## Commands Used for Training and Testing
1. **TF-IDF Vectorization**:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   tfidf_vectorizer = TfidfVectorizer()
   tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
   ```

2. **Training Random Forest Classifiers**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf_classifier = RandomForestClassifier()
   rf_classifier.fit(X_train, y_train)
   ```

3. **Saving Models**:
   ```python
   import pickle
   with open('model_name.pkl', 'wb') as file:
       pickle.dump(model, file)
   ```

4. **Loading Models**:
   ```python
   with open('model_name.pkl', 'rb') as file:
       model = pickle.load(file)
   ```

5. **Evaluating the Model**:
   ```python
   from sklearn.metrics import accuracy_score
   predictions = rf_classifier.predict(X_test)
   accuracy = accuracy_score(y_test, predictions)
   print(f'Accuracy: {accuracy * 100:.2f}%')
   ```

---

## Troubleshooting
1. **Invalid File Format**:
   - Ensure the uploaded resume is in PDF or TXT format.
2. **Model Not Found**:
   - Verify that all required model files are in the `models/` directory.
3. **Port Already in Use**:
   - Modify the `app.run()` command in `app.py` to specify a different port:
     ```python
     app.run(port=5001)
     ```

## Acknowledgments
- **Datasets**: [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset), [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset).
- **Libraries**: Flask, Scikit-learn, PyPDF2.


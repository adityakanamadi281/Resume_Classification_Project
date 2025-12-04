import pickle
import re

with open('tfidf_with_skills.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model_with_skills.pkl', 'rb') as f:
    model = pickle.load(f)

skill_keywords = {
    'React Developer': ['react', 'javascript', 'js', 'jsx', 'redux', 'html', 'css', 'node', 'typescript', 'npm', 'webpack', 'frontend', 'ui', 'ux'],
    'workday resumes': ['workday', 'hcm', 'hrms', 'payroll', 'benefits', 'recruiting', 'talent', 'integration', 'studio', 'eib'],
    'SQL Developer Lightning insight': ['sql', 'mysql', 'postgresql', 'oracle', 'database', 'query', 'stored procedure', 'etl', 'ssrs', 'ssis', 'plsql'],
    'Peoplesoft resumes': ['peoplesoft', 'oracle', 'fscm', 'hrms', 'peopletools', 'sqr', 'app engine', 'peoplecode']
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills_from_text(text):
    text_lower = text.lower()
    all_skills = []
    for category, skills in skill_keywords.items():
        for skill in skills:
            if skill in text_lower:
                all_skills.append(skill)
    return ', '.join(set(all_skills)) if all_skills else ''

def classify_resume(resume_text):
    cleaned = clean_text(resume_text)
    skills = extract_skills_from_text(resume_text)
    combined = cleaned + ' ' + skills
    vectorized = tfidf.transform([combined])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    return prediction, probabilities, skills

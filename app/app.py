import streamlit as st
import pickle
import re
import PyPDF2
from docx import Document
import docx2txt
import io
import pandas as pd

# THIS MUST BE FIRST!
st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")

# Load model
@st.cache_resource
def load_model():
    with open(r'C:\Users\adity\Resume_Classification_Project\models\tfidf_with_skills.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open(r'C:\Users\adity\Resume_Classification_Project\models\model_with_skills.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model


tfidf, model = load_model()

# Skills keywords
skill_keywords = {
    'React Developer': ['react', 'javascript', 'js', 'jsx', 'redux', 'html', 'css', 'node', 'typescript', 'npm',
                        'webpack', 'frontend', 'ui', 'ux'],
    'workday resumes': ['workday', 'hcm', 'hrms', 'payroll', 'benefits', 'recruiting', 'talent', 'integration',
                        'studio', 'eib'],
    'SQL Developer Lightning insight': ['sql', 'mysql', 'postgresql', 'oracle', 'database', 'query', 'stored procedure',
                                        'etl', 'ssrs', 'ssis', 'plsql'],
    'Peoplesoft resumes': ['peoplesoft', 'oracle', 'fscm', 'hrms', 'peopletools', 'sqr', 'app engine', 'peoplecode']
}


def read_pdf(file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def read_docx(file):
    """Extract text from DOCX"""
    try:
        doc = Document(file)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None


def read_doc(file):
    """Extract text from DOC"""
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error reading DOC: {e}")
        return None


def clean_text(text):
    """Clean resume text"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills_from_text(text):
    """Extract skills from text"""
    text_lower = text.lower()
    all_skills = []

    for category, skills in skill_keywords.items():
        for skill in skills:
            if skill in text_lower:
                all_skills.append(skill)

    return list(set(all_skills))


def classify_resume(resume_text):
    """Classify resume"""
    cleaned = clean_text(resume_text)
    skills = extract_skills_from_text(resume_text)
    skills_str = ', '.join(skills) if skills else ''

    combined = cleaned + ' ' + skills_str
    vectorized = tfidf.transform([combined])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    return prediction, probabilities, skills


# Streamlit UI
st.title("📄 Resume Classification System")
st.markdown("### Upload a resume to classify it into job categories")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This system classifies resumes into:
    - React Developer
    - Workday Resumes
    - SQL Developer
    - PeopleSoft Resumes

    **Model Accuracy: 100%**
    """)

    st.header("Supported Formats")
    st.write("- PDF (.pdf)")
    st.write("- Word Document (.docx)")
    st.write("- Old Word (.doc)")

# Main content
uploaded_file = st.file_uploader("Choose a resume file", type=['pdf', 'docx', 'doc'])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    # Extract text based on file type
    file_extension = uploaded_file.name.split('.')[-1].lower()

    with st.spinner("Reading resume..."):
        if file_extension == 'pdf':
            resume_text = read_pdf(uploaded_file)
        elif file_extension == 'docx':
            resume_text = read_docx(uploaded_file)
        elif file_extension == 'doc':
            resume_text = read_doc(uploaded_file)
        else:
            st.error("Unsupported file format!")
            resume_text = None

    if resume_text:
        # Show resume preview
        with st.expander("📄 View Resume Content (First 500 characters)"):
            st.text(resume_text[:500] + "...")

        # Classify
        with st.spinner("Classifying resume..."):
            category, probabilities, skills = classify_resume(resume_text)

        # Display results
        st.markdown("---")
        st.header("🎯 Classification Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted Category")
            st.success(f"## {category}")

            confidence = max(probabilities) * 100
            st.metric("Confidence", f"{confidence:.2f}%")

        with col2:
            st.subheader("Extracted Skills")
            if skills:
                skills_html = ""
                for skill in skills:
                    skills_html += f'<span style="background-color: #0066cc; color: white; padding: 5px 10px; margin: 5px; border-radius: 15px; display: inline-block;">{skill}</span> '
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.warning("No specific skills detected")

        # Show all probabilities
        st.markdown("---")
        st.subheader("📊 Confidence Scores for All Categories")

        prob_data = []
        for cat, prob in zip(model.classes_, probabilities):
            prob_data.append({"Category": cat, "Probability": prob * 100})

        prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)

        st.bar_chart(prob_df.set_index('Category'))

        # Show detailed probabilities
        for idx, row in prob_df.iterrows():
            st.progress(row['Probability'] / 100, text=f"{row['Category']}: {row['Probability']:.2f}%")

else:
    st.info("👆 Please upload a resume file to get started")

    # Example section
    st.markdown("---")
    st.subheader("💡 Try with sample text")

    sample_text = st.text_area("Or paste resume text here:", height=200)

    if st.button("Classify Text"):
        if sample_text:
            category, probabilities, skills = classify_resume(sample_text)

            st.success(f"**Predicted Category:** {category}")
            st.info(f"**Confidence:** {max(probabilities) * 100:.2f}%")
            if skills:
                st.write("**Skills Found:**", ', '.join(skills))
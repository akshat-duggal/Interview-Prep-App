import streamlit as st
import PyPDF2
import pdfplumber
from groq import Groq
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json
import re
from typing import Dict, List, Optional

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            return text
        except:
            return ""
    return text

def extract_email(text: str) -> str:
    """Extract email address from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not found"

def extract_phone(text: str) -> str:
    """Extract phone number from text."""
    phone_patterns = [
        r'\+?\d{1,3}[-\.\s]?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}',
        r'\+?\d{10,}',
        r'\d{3}-\d{3}-\d{4}',
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            return phones[0]
    return "Not found"

def extract_skills(text: str) -> List[str]:
    """Extract technical skills from resume text."""
    skills_database = [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
        'kotlin', 'go', 'rust', 'typescript', 'scala', 'r', 'matlab',
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
        'django', 'flask', 'spring boot', 'asp.net', 'jquery',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
        'sqlite', 'cassandra', 'dynamodb', 'firebase',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
        'ci/cd', 'terraform', 'ansible', 'linux', 'shell scripting',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy', 'data analysis', 'statistics',
        'nlp', 'computer vision', 'opencv',
        'agile', 'scrum', 'jira', 'rest api', 'graphql', 'microservices',
        'testing', 'debugging', 'problem solving', 'communication'
    ]
    text_lower = text.lower()
    found_skills = []
    for skill in skills_database:
        if skill in text_lower:
            found_skills.append(skill.title())
    return sorted(list(set(found_skills)))

def extract_education(text: str) -> List[str]:
    """Extract education information."""
    education_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 
                         'b.e', 'm.e', 'bca', 'mca', 'degree', 'university',
                         'college', 'institute', 'b.sc', 'm.sc']
    lines = text.split('\n')
    education = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in education_keywords):
            edu_text = ' '.join(lines[i:i+3]).strip()
            if edu_text and len(edu_text) > 10:
                education.append(edu_text[:200])
    return education[:3]

def extract_experience(text: str) -> List[Dict[str, str]]:
    """Extract work experience details."""
    experience = []
    job_keywords = ['engineer', 'developer', 'analyst', 'manager', 'designer',
                   'consultant', 'intern', 'associate', 'specialist', 'lead']
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in job_keywords):
            exp_entry = {
                'title': line.strip()[:100],
                'description': ' '.join(lines[i+1:i+4]).strip()[:300]
            }
            experience.append(exp_entry)
    return experience[:5]

def structure_resume_data(text: str) -> Dict:
    """Main function to structure all extracted resume data."""
    structured_data = {
        'raw_text': text,
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'education': extract_education(text),
        'experience': extract_experience(text),
        'total_words': len(text.split()),
        'total_characters': len(text)
    }
    return structured_data

def calculate_ats_score(resume_data: Dict) -> Dict:
    """Calculate ATS compatibility score."""
    score = 0
    feedback = []
    
    if resume_data['email'] != "Not found":
        score += 8
        feedback.append("✅ Email found")
    else:
        feedback.append("❌ Missing email address")
    
    if resume_data['phone'] != "Not found":
        score += 7
        feedback.append("✅ Phone number found")
    else:
        feedback.append("❌ Missing phone number")
    
    num_skills = len(resume_data['skills'])
    if num_skills >= 10:
        score += 30
        feedback.append(f"✅ Strong skills section ({num_skills} skills)")
    elif num_skills >= 5:
        score += 20
        feedback.append(f"⚠️ Moderate skills section ({num_skills} skills)")
    else:
        score += 10
        feedback.append(f"❌ Weak skills section ({num_skills} skills)")
    
    if len(resume_data['education']) >= 1:
        score += 15
        feedback.append("✅ Education section present")
    else:
        feedback.append("❌ Missing education section")
    
    num_exp = len(resume_data['experience'])
    if num_exp >= 3:
        score += 25
        feedback.append(f"✅ Strong experience section ({num_exp} entries)")
    elif num_exp >= 1:
        score += 15
        feedback.append(f"⚠️ Limited experience ({num_exp} entries)")
    else:
        feedback.append("❌ No experience section found")
    
    word_count = resume_data['total_words']
    if 300 <= word_count <= 800:
        score += 15
        feedback.append(f"✅ Optimal length ({word_count} words)")
    elif word_count < 300:
        score += 5
        feedback.append(f"⚠️ Too short ({word_count} words)")
    else:
        score += 10
        feedback.append(f"⚠️ Too long ({word_count} words)")
    
    return {
        'score': score,
        'grade': 'A+' if score >= 90 else 'A' if score >= 80 else 'B' if score >= 70 else 'C' if score >= 60 else 'D',
        'feedback': feedback
    }

def analyze_resume_with_gemini(resume_data: Dict, gemini_model) -> Dict:
    """Use Gemini AI to provide deep analysis."""
    prompt = f"""You are an expert resume reviewer. Analyze this resume and provide detailed feedback.

RESUME DATA:
- Total Words: {resume_data['total_words']}
- Skills ({len(resume_data['skills'])}): {', '.join(resume_data['skills'][:15])}
- Education Entries: {len(resume_data['education'])}
- Experience Entries: {len(resume_data['experience'])}

RESUME TEXT:
{resume_data['raw_text'][:3000]}

Provide:
1. STRENGTHS (3-4 points)
2. WEAKNESSES (3-4 points)
3. MISSING KEYWORDS
4. IMPROVEMENTS (5 suggestions)
5. OVERALL IMPRESSION

Format clearly with headers."""

    try:
        response = gemini_model.generate_content(prompt)
        return {'success': True, 'analysis': response.text}
    except Exception as e:
        return {'success': False, 'analysis': "Analysis unavailable."}

def generate_improvement_suggestions(resume_data: Dict, ats_score: Dict) -> List[str]:
    """Generate improvement suggestions."""
    suggestions = []
    
    if resume_data['email'] == "Not found":
        suggestions.append("🔸 Add a professional email address")
    if resume_data['phone'] == "Not found":
        suggestions.append("🔸 Include a contact phone number")
    if len(resume_data['skills']) < 8:
        suggestions.append("🔸 Add more technical skills (aim for 10-15)")
    if len(resume_data['education']) == 0:
        suggestions.append("🔸 Add Education section")
    if len(resume_data['experience']) < 2:
        suggestions.append("🔸 Add more work experience details")
    if resume_data['total_words'] < 300:
        suggestions.append("🔸 Expand with more details")
    
    suggestions.append("🔸 Quantify achievements with numbers")
    suggestions.append("🔸 Tailor resume for each job")
    suggestions.append("🔸 Use industry-specific keywords")
    
    return suggestions[:8]

def complete_resume_analysis(resume_data: Dict, gemini_model, groq_client) -> Dict:
    """Complete resume analysis."""
    ats_result = calculate_ats_score(resume_data)
    gemini_analysis = analyze_resume_with_gemini(resume_data, gemini_model)
    suggestions = generate_improvement_suggestions(resume_data, ats_result)
    
    return {
        'ats_score': ats_result['score'],
        'ats_grade': ats_result['grade'],
        'ats_feedback': ats_result['feedback'],
        'gemini_analysis': gemini_analysis['analysis'],
        'improvement_suggestions': suggestions,
        'contact_info': {
            'email': resume_data['email'],
            'phone': resume_data['phone']
        },
        'stats': {
            'total_skills': len(resume_data['skills']),
            'total_experience': len(resume_data['experience']),
            'total_education': len(resume_data['education']),
            'word_count': resume_data['total_words']
        }
    }

def format_score_badge(score: float) -> str:
    """Generate score badge."""
    if score >= 90:
        return "🏆 Excellent"
    elif score >= 80:
        return "⭐ Very Good"
    elif score >= 70:
        return "✅ Good"
    elif score >= 60:
        return "👍 Fair"
    elif score >= 50:
        return "📈 Needs Work"
    else:
        return "⚠️ More Practice Needed"

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="AI Interview Prep & Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .score-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .score-value {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .info-box {
        background: #f3f4f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    .info-box.success {background: #ecfdf5; border-left-color: #10b981;}
    .info-box.warning {background: #fffbeb; border-left-color: #f59e0b;}
    .info-box.error {background: #fef2f2; border-left-color: #ef4444;}
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-info {background: #dbeafe; color: #1e40af;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "api_keys_set" not in st.session_state:
    st.session_state.api_keys_set = False

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    if not st.session_state.api_keys_set:
        groq_key = st.text_input("Groq API Key", type="password", key="groq_key_input")
        gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
        
        if st.button("✅ Save API Keys"):
            if groq_key and gemini_key:
                try:
                    # Initialize clients
                    st.session_state.groq_client = Groq(api_key=groq_key)
                    genai.configure(api_key=gemini_key)
                    st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    
                    # Test connections
                    test = st.session_state.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": "Hi"}],
                        model="llama-3.3-70b-versatile",
                        max_tokens=5
                    )
                    
                    st.session_state.api_keys_set = True
                    st.success("✅ Connected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.error("Enter both keys")
    else:
        st.success("✅ APIs Connected")
        if st.button("🔄 Reset"):
            st.session_state.api_keys_set = False
            st.rerun()
    
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "📄 Resume Analysis"], label_visibility="collapsed")

# Main content
if not st.session_state.api_keys_set:
    st.markdown('<div class="main-header"><h1>🎯 AI Interview Prep & Resume Analyzer</h1><p>Enter API keys to start</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔑 Groq API\n1. Visit [console.groq.com](https://console.groq.com/keys)\n2. Create API Key\n3. Paste in sidebar")
    with col2:
        st.markdown("### 🔑 Gemini API\n1. Visit [aistudio.google.com](https://aistudio.google.com/app/apikey)\n2. Create API Key\n3. Paste in sidebar")

else:
    if page == "🏠 Home":
        st.markdown('<div class="main-header"><h1>🎯 AI Interview Prep</h1><p>Your AI career assistant</p></div>', unsafe_allow_html=True)
        
        if st.session_state.resume_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><div style="font-size:2rem;font-weight:700;color:#667eea">{len(st.session_state.resume_data.get("skills", []))}</div><div>Skills Found</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div style="font-size:2rem;font-weight:700;color:#667eea">{len(st.session_state.resume_data.get("experience", []))}</div><div>Experience</div></div>', unsafe_allow_html=True)
            with col3:
                if st.session_state.resume_analysis:
                    score = st.session_state.resume_analysis['ats_score']
                    st.markdown(f'<div class="metric-card"><div style="font-size:2rem;font-weight:700;color:#667eea">{score}</div><div>ATS Score</div></div>', unsafe_allow_html=True)
        else:
            st.info("👆 Upload resume in sidebar")
    
    elif page == "📄 Resume Analysis":
        st.markdown('<div class="main-header"><h1>📄 Resume Analysis</h1><p>Upload PDF for analysis</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'])
        
        if uploaded_file and st.button("🔍 Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    resume_data = structure_resume_data(text)
                    st.session_state.resume_data = resume_data
                    
                    analysis = complete_resume_analysis(
                        resume_data, 
                        st.session_state.gemini_model,
                        st.session_state.groq_client
                    )
                    st.session_state.resume_analysis = analysis
                    st.success("✅ Complete!")
                    st.rerun()
        
        if st.session_state.resume_analysis:
            st.markdown("## 📊 ATS Score")
            
            col1, col2, col3 = st.columns([2,1,1])
            with col1:
                score = st.session_state.resume_analysis['ats_score']
                st.markdown(f'<div class="score-card"><p class="score-value">{score}</p><p>ATS Score / 100</p></div>', unsafe_allow_html=True)
            with col2:
                grade = st.session_state.resume_analysis['ats_grade']
                st.markdown(f'<div class="metric-card"><div style="font-size:2rem;font-weight:700;color:#667eea">{grade}</div><div>Grade</div></div>', unsafe_allow_html=True)
            with col3:
                badge = format_score_badge(score)
                st.markdown(f'<div class="info-box success"><h3 style="margin:0">{badge}</h3></div>', unsafe_allow_html=True)
            
            st.markdown("### 📋 Feedback")
            for feedback in st.session_state.resume_analysis['ats_feedback']:
                if '✅' in feedback:
                    st.markdown(f'<div class="info-box success">{feedback}</div>', unsafe_allow_html=True)
                elif '⚠️' in feedback:
                    st.markdown(f'<div class="info-box warning">{feedback}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-box error">{feedback}</div>', unsafe_allow_html=True)
            
            st.markdown("## 🤖 AI Analysis")
            with st.expander("View Details", expanded=True):
                st.markdown(st.session_state.resume_analysis['gemini_analysis'])
            
            st.markdown("## 💡 Suggestions")
            for i, suggestion in enumerate(st.session_state.resume_analysis['improvement_suggestions'], 1):
                st.markdown(f"{i}. {suggestion}")
            
            st.markdown("## 🎯 Skills")
            skills = st.session_state.resume_data['skills']
            skills_html = '<div>'
            for skill in skills:
                skills_html += f'<span class="badge badge-info">{skill}</span>'
            skills_html += '</div>'
            st.markdown(skills_html, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#6b7280;padding:1rem"><p>🎯 AI Interview Prep v1.0</p></div>', unsafe_allow_html=True)

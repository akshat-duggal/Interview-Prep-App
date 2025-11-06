import streamlit as st
import PyPDF2
import pdfplumber
from groq import Groq
import google.generativeai as genai
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import re
from typing import Dict, List

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except:
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except:
            pass
    return text

def extract_email(text: str) -> str:
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails[0] if emails else "Not found"

def extract_phone(text: str) -> str:
    patterns = [r'\+?\d{1,3}[-\.\s]?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}', r'\+?\d{10,}']
    for pattern in patterns:
        phones = re.findall(pattern, text)
        if phones:
            return phones[0]
    return "Not found"

def extract_skills(text: str) -> List[str]:
    skills_db = ['python', 'java', 'javascript', 'c++', 'c#', 'react', 'angular', 'vue', 'node.js',
                 'django', 'flask', 'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure', 'docker',
                 'kubernetes', 'git', 'machine learning', 'data analysis', 'agile', 'scrum']
    found = [s.title() for s in skills_db if s in text.lower()]
    return sorted(list(set(found)))

def extract_education(text: str) -> List[str]:
    keywords = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'degree', 'university']
    lines = text.split('\n')
    education = []
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in keywords):
            edu = ' '.join(lines[i:i+2]).strip()[:150]
            if len(edu) > 10:
                education.append(edu)
    return education[:3]

def extract_experience(text: str) -> List[Dict]:
    keywords = ['engineer', 'developer', 'analyst', 'manager', 'intern']
    lines = text.split('\n')
    exp = []
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in keywords):
            exp.append({'title': line.strip()[:80], 'description': ' '.join(lines[i+1:i+3]).strip()[:200]})
    return exp[:5]

def structure_resume_data(text: str) -> Dict:
    return {
        'raw_text': text,
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'education': extract_education(text),
        'experience': extract_experience(text),
        'total_words': len(text.split())
    }

def calculate_ats_score(data: Dict) -> Dict:
    score = 0
    feedback = []
    
    if data['email'] != "Not found":
        score += 10
        feedback.append("✅ Email found")
    else:
        feedback.append("❌ Missing email")
    
    if data['phone'] != "Not found":
        score += 10
        feedback.append("✅ Phone found")
    else:
        feedback.append("❌ Missing phone")
    
    skills = len(data['skills'])
    if skills >= 10:
        score += 30
        feedback.append(f"✅ Strong skills ({skills})")
    elif skills >= 5:
        score += 20
        feedback.append(f"⚠️ Moderate skills ({skills})")
    else:
        score += 10
        feedback.append(f"❌ Weak skills ({skills})")
    
    if data['education']:
        score += 20
        feedback.append("✅ Education present")
    else:
        feedback.append("❌ Missing education")
    
    exp = len(data['experience'])
    if exp >= 3:
        score += 30
        feedback.append(f"✅ Strong experience ({exp})")
    elif exp >= 1:
        score += 15
        feedback.append(f"⚠️ Limited experience ({exp})")
    else:
        feedback.append("❌ No experience")
    
    return {'score': score, 'grade': 'A' if score >= 80 else 'B' if score >= 60 else 'C', 'feedback': feedback}

def analyze_with_gemini(data: Dict, model) -> str:
    prompt = f"""Analyze resume: {data['total_words']} words, {len(data['skills'])} skills, {len(data['experience'])} roles.

Provide: 1) Strengths 2) Weaknesses 3) Improvements"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI analysis unavailable"

def generate_technical_questions(data: Dict, difficulty: str, num: int, client) -> List[Dict]:
    skills = ', '.join(data['skills'][:8])
    prompt = f"Generate {num} {difficulty} technical questions for skills: {skills}. Format: Q1: [question]"
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=1000
        )
        text = response.choices[0].message.content
        questions = []
        for i, block in enumerate(text.split('\n\n')[:num], 1):
            if block.strip():
                questions.append({'id': i, 'type': 'Technical', 'difficulty': difficulty, 'question': block.strip()})
        return questions
    except:
        return [{'id': i, 'type': 'Technical', 'difficulty': difficulty, 'question': f'Sample question {i}'} for i in range(1, num+1)]

def generate_behavioral_questions(num: int, client) -> List[Dict]:
    prompt = f"Generate {num} behavioral interview questions. Format: Q1: [question]"
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=800
        )
        text = response.choices[0].message.content
        questions = []
        for i, block in enumerate(text.split('\n\n')[:num], 1):
            if block.strip():
                questions.append({'id': i, 'type': 'Behavioral', 'difficulty': 'Medium', 'question': block.strip()})
        return questions
    except:
        return [{'id': i, 'type': 'Behavioral', 'difficulty': 'Medium', 'question': f'Sample behavioral question {i}'} for i in range(1, num+1)]

def evaluate_answer(question: str, answer: str, qtype: str, client) -> Dict:
    prompt = f"Evaluate answer. Question: {question}\nAnswer: {answer}\n\nScore 0-10 for: relevance, accuracy, communication. Provide feedback."
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=500
        )
        text = response.choices[0].message.content
        
        rel = re.search(r'relevance.*?(\d+)', text.lower())
        acc = re.search(r'accuracy.*?(\d+)', text.lower())
        comm = re.search(r'communication.*?(\d+)', text.lower())
        
        scores = {
            'relevance': int(rel.group(1)) if rel else 7,
            'accuracy': int(acc.group(1)) if acc else 7,
            'communication': int(comm.group(1)) if comm else 7
        }
        scores['total'] = sum(scores.values())
        scores['percentage'] = round((scores['total'] / 30) * 100, 1)
        
        return {'success': True, 'scores': scores, 'feedback': text}
    except:
        return {'success': False, 'scores': {'total': 21, 'percentage': 70}, 'feedback': 'Evaluation unavailable'}

def extract_job_requirements(jd: str) -> Dict:
    skills_db = ['python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker']
    required = [s.title() for s in skills_db if s in jd.lower()]
    
    exp_match = re.search(r'(\d+)\+?\s*(?:year|yr)', jd.lower())
    exp_required = int(exp_match.group(1)) if exp_match else 0
    
    return {'required_skills': required, 'required_experience': exp_required}

def calculate_match_score(resume: Dict, job_req: Dict) -> Dict:
    score = 0
    
    resume_skills = set([s.lower() for s in resume['skills']])
    required_skills = set([s.lower() for s in job_req['required_skills']])
    
    if required_skills:
        matching = resume_skills & required_skills
        missing = required_skills - resume_skills
        match_pct = len(matching) / len(required_skills)
        skill_points = int(match_pct * 60)
        score += skill_points
        breakdown = {'skills': {'score': skill_points, 'matching': list(matching), 'missing': list(missing)}}
    else:
        score += 30
        breakdown = {'skills': {'score': 30, 'matching': [], 'missing': []}}
    
    exp = len(resume['experience'])
    if exp >= job_req['required_experience']:
        score += 25
    else:
        score += 15
    
    if resume['education']:
        score += 15
    
    return {'total_score': score, 'percentage': score, 'breakdown': breakdown, 
            'grade': 'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Fair'}

def generate_cover_letter(resume: Dict, jd: str, client) -> str:
    skills = ', '.join(resume['skills'][:8])
    prompt = f"Write 200-word cover letter. Skills: {skills}. Job: {jd[:500]}"
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=400
        )
        return response.choices[0].message.content
    except:
        return "Cover letter generation unavailable"

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="AI Interview Prep", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.main {padding: 2rem;}
.header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; 
         border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;}
.score-card {background: white; padding: 1.5rem; border-radius: 12px; text-align: center; 
             box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #667eea;}
.metric {background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);}
.info {background: #f3f4f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #3b82f6;}
.success {background: #ecfdf5; border-left-color: #10b981;}
.warning {background: #fffbeb; border-left-color: #f59e0b;}
.error {background: #fef2f2; border-left-color: #ef4444;}
.badge {display: inline-block; padding: 0.4rem 0.8rem; border-radius: 20px; 
        font-size: 0.85rem; margin: 0.2rem; background: #dbeafe; color: #1e40af;}
</style>
""", unsafe_allow_html=True)

# Session state
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "job_match" not in st.session_state:
    st.session_state.job_match = None
if "current_interview" not in st.session_state:
    st.session_state.current_interview = None
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "api_set" not in st.session_state:
    st.session_state.api_set = False

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 API Keys")
    
    if not st.session_state.api_set:
        groq_key = st.text_input("Groq API Key", type="password")
        gemini_key = st.text_input("Gemini API Key", type="password")
        
        if st.button("Save Keys"):
            if groq_key and gemini_key:
                try:
                    st.session_state.groq_client = Groq(api_key=groq_key)
                    genai.configure(api_key=gemini_key)
                    st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    
                    # Test
                    st.session_state.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": "test"}],
                        model="llama-3.3-70b-versatile",
                        max_tokens=5
                    )
                    
                    st.session_state.api_set = True
                    st.success("✅ Connected")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.success("✅ APIs Active")
        if st.button("Reset"):
            st.session_state.api_set = False
            st.rerun()
    
    st.markdown("---")
    page = st.radio("Menu", ["🏠 Home", "📄 Resume", "🎯 Job Match", "🎤 Interview", "📊 Dashboard"])
    
    if st.session_state.resume_data:
        st.markdown("---")
        st.metric("Skills", len(st.session_state.resume_data['skills']))
        if st.session_state.interview_history:
            st.metric("Interviews", len(st.session_state.interview_history))

# Main
if not st.session_state.api_set:
    st.markdown('<div class="header"><h1>🎯 AI Interview Prep</h1><p>Enter API keys to start</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Groq\n1. [console.groq.com](https://console.groq.com/keys)\n2. Create key\n3. Paste left")
    with col2:
        st.markdown("### Gemini\n1. [aistudio.google.com](https://aistudio.google.com/app/apikey)\n2. Create key\n3. Paste left")

else:
    if page == "🏠 Home":
        st.markdown('<div class="header"><h1>🎯 AI Interview Prep</h1><p>Your AI career assistant</p></div>', unsafe_allow_html=True)
        
        if st.session_state.resume_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric"><h2>{len(st.session_state.resume_data["skills"])}</h2><p>Skills</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric"><h2>{len(st.session_state.resume_data["experience"])}</h2><p>Roles</p></div>', unsafe_allow_html=True)
            with col3:
                if st.session_state.resume_analysis:
                    st.markdown(f'<div class="metric"><h2>{st.session_state.resume_analysis["ats_score"]}</h2><p>ATS Score</p></div>', unsafe_allow_html=True)
        else:
            st.info("Upload resume in 📄 Resume tab")
    
    elif page == "📄 Resume":
        st.markdown('<div class="header"><h1>📄 Resume Analysis</h1></div>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded and st.button("Analyze", type="primary"):
            with st.spinner("Processing..."):
                text = extract_text_from_pdf(uploaded)
                if text:
                    data = structure_resume_data(text)
                    st.session_state.resume_data = data
                    
                    ats = calculate_ats_score(data)
                    ai_analysis = analyze_with_gemini(data, st.session_state.gemini_model)
                    
                    st.session_state.resume_analysis = {
                        'ats_score': ats['score'],
                        'ats_grade': ats['grade'],
                        'feedback': ats['feedback'],
                        'ai_analysis': ai_analysis
                    }
                    st.success("Done!")
                    st.rerun()
        
        if st.session_state.resume_analysis:
            st.markdown("## ATS Score")
            col1, col2 = st.columns([2,1])
            with col1:
                score = st.session_state.resume_analysis['ats_score']
                st.markdown(f'<div class="score-card"><h1 style="font-size:3rem;margin:0">{score}</h1><p>/ 100</p></div>', unsafe_allow_html=True)
            with col2:
                grade = st.session_state.resume_analysis['ats_grade']
                st.markdown(f'<div class="metric"><h2>{grade}</h2><p>Grade</p></div>', unsafe_allow_html=True)
            
            st.markdown("### Feedback")
            for f in st.session_state.resume_analysis['feedback']:
                cls = 'success' if '✅' in f else 'warning' if '⚠️' in f else 'error'
                st.markdown(f'<div class="info {cls}">{f}</div>', unsafe_allow_html=True)
            
            st.markdown("### AI Analysis")
            with st.expander("View", expanded=True):
                st.write(st.session_state.resume_analysis['ai_analysis'])
            
            st.markdown("### Skills")
            html = ''
            for s in st.session_state.resume_data['skills']:
                html += f'<span class="badge">{s}</span>'
            st.markdown(html, unsafe_allow_html=True)
    
    elif page == "🎯 Job Match":
        st.markdown('<div class="header"><h1>🎯 Job Matcher</h1></div>', unsafe_allow_html=True)
        
        if not st.session_state.resume_data:
            st.warning("Upload resume first")
        else:
            jd = st.text_area("Job Description", height=200)
            
            if jd and st.button("Match", type="primary"):
                with st.spinner("Analyzing..."):
                    job_req = extract_job_requirements(jd)
                    match = calculate_match_score(st.session_state.resume_data, job_req)
                    cover = generate_cover_letter(st.session_state.resume_data, jd, st.session_state.groq_client)
                    
                    st.session_state.job_match = {
                        'match_score': match['percentage'],
                        'grade': match['grade'],
                        'breakdown': match['breakdown'],
                        'cover_letter': cover
                    }
                    st.success("Done!")
                    st.rerun()
            
            if st.session_state.job_match:
                st.markdown("## Match Score")
                col1, col2 = st.columns([2,1])
                with col1:
                    score = st.session_state.job_match['match_score']
                    st.markdown(f'<div class="score-card"><h1 style="font-size:3rem;margin:0">{score}%</h1></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric"><h3>{st.session_state.job_match["grade"]}</h3></div>', unsafe_allow_html=True)
                
                st.markdown("### Skills")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Matching**")
                    for s in st.session_state.job_match['breakdown']['skills']['matching']:
                        st.markdown(f'<span class="badge">{s}</span>', unsafe_allow_html=True)
                with col2:
                    st.markdown("**Missing**")
                    for s in st.session_state.job_match['breakdown']['skills']['missing']:
                        st.markdown(f'<span class="badge" style="background:#fee2e2;color:#991b1b">{s}</span>', unsafe_allow_html=True)
                
                st.markdown("### Cover Letter")
                st.text_area("", st.session_state.job_match['cover_letter'], height=300)
                st.download_button("Download", st.session_state.job_match['cover_letter'], "cover.txt")
    
    elif page == "🎤 Interview":
        st.markdown('<div class="header"><h1>🎤 Mock Interview</h1></div>', unsafe_allow_html=True)
        
        if not st.session_state.resume_data:
            st.warning("Upload resume first")
        else:
            if not st.session_state.current_interview:
                col1, col2 = st.columns(2)
                with col1:
                    itype = st.selectbox("Type", ["Technical", "Behavioral", "Mixed"])
                with col2:
                    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
                
                if st.button("Generate Questions", type="primary"):
                    with st.spinner("Generating..."):
                        if itype == "Technical":
                            questions = generate_technical_questions(st.session_state.resume_data, diff, 5, st.session_state.groq_client)
                        elif itype == "Behavioral":
                            questions = generate_behavioral_questions(5, st.session_state.groq_client)
                        else:
                            questions = generate_technical_questions(st.session_state.resume_data, diff, 3, st.session_state.groq_client)
                            questions += generate_behavioral_questions(2, st.session_state.groq_client)
                        
                        st.session_state.current_interview = {
                            'questions': questions,
                            'answers': [],
                            'current': 0,
                            'type': itype
                        }
                        st.success(f"Generated {len(questions)} questions")
                        st.rerun()
            else:
                questions = st.session_state.current_interview['questions']
                current = st.session_state.current_interview['current']
                
                st.progress(current / len(questions))
                st.write(f"Question {current + 1} of {len(questions)}")
                
                if current < len(questions):
                    q = questions[current]
                    st.markdown(f'<div class="info"><b>Q{current+1}:</b> {q["question"]}</div>', unsafe_allow_html=True)
                    
                    answer = st.text_area("Your Answer", height=150, key=f"ans_{current}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Skip"):
                            st.session_state.current_interview['answers'].append("")
                            st.session_state.current_interview['current'] += 1
                            st.rerun()
                    with col2:
                        if st.button("Submit", type="primary", disabled=not answer):
                            st.session_state.current_interview['answers'].append(answer)
                            st.session_state.current_interview['current'] += 1
                            st.rerun()
                else:
                    st.success("Interview Complete!")
                    
                    if st.button("View Results", type="primary"):
                        with st.spinner("Evaluating..."):
                            results = []
                            total = 0
                            
                            for q, a in zip(questions, st.session_state.current_interview['answers']):
                                if a:
                                    eval_result = evaluate_answer(q['question'], a, q['type'], st.session_state.groq_client)
                                    results.append({
                                        'question': q['question'],
                                        'answer': a,
                                        'scores': eval_result['scores'],
                                        'feedback': eval_result['feedback']
                                    })
                                    total += eval_result['scores']['percentage']
                            
                            avg = total / len(results) if results else 0
                            
                            interview_record = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'type': st.session_state.current_interview['type'],
                                'questions': len(questions),
                                'score': avg,
                                'results': results
                            }
                            
                            st.session_state.interview_history.append(interview_record)
                            st.session_state.interview_results = interview_record
                            st.session_state.current_interview = None
                            st.rerun()
            
            if 'interview_results' in st.session_state and st.session_state.interview_results:
                st.markdown("---")
                st.markdown("## Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    score = st.session_state.interview_results['score']
                    st.markdown(f'<div class="score-card"><h1 style="font-size:3rem;margin:0">{score:.1f}%</h1></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric"><h3>{st.session_state.interview_results["questions"]} Questions</h3></div>', unsafe_allow_html=True)
                
                st.markdown("### Details")
                for i, r in enumerate(st.session_state.interview_results['results'], 1):
                    with st.expander(f"Q{i}: {r['scores']['percentage']:.0f}%"):
                        st.write(f"**Q:** {r['question']}")
                        st.write(f"**A:** {r['answer']}")
                        st.write(f"**Scores:** Rel {r['scores']['relevance']}/10, Acc {r['scores']['accuracy']}/10, Comm {r['scores']['communication']}/10")
                        st.write(f"**Feedback:** {r['feedback'][:200]}...")
                
                if st.button("New Interview"):
                    st.session_state.interview_results = None
                    st.rerun()
    
    elif page == "📊 Dashboard":
        st.markdown('<div class="header"><h1>📊 Dashboard</h1></div>', unsafe_allow_html=True)
        
        if not st.session_state.interview_history:
            st.info("Complete interviews to see stats")
        else:
            total = len(st.session_state.interview_history)
            avg_score = sum(i['score'] for i in st.session_state.interview_history) / total
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric"><h2>{total}</h2><p>Interviews</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric"><h2>{avg_score:.1f}%</h2><p>Avg Score</p></div>', unsafe_allow_html=True)
            
            st.markdown("### Score Trend")
            scores = [i['score'] for i in st.session_state.interview_history]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(scores)+1)), y=scores, mode='lines+markers'))
            fig.update_layout(xaxis_title="Session", yaxis_title="Score (%)", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### History")
            data = []
            for i in st.session_state.interview_history:
                data.append({
                    'Date': i['timestamp'],
                    'Type': i['type'],
                    'Questions': i['questions'],
                    'Score': f"{i['score']:.1f}%"
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#6b7280"><p>🎯 AI Interview Prep v1.0</p></div>', unsafe_allow_html=True)

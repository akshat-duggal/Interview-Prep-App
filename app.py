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
# GROQ CLIENT HELPER
# ============================================================================

def get_groq_client():
    """Get or create Groq client with lazy initialization."""
    if 'groq_client' not in st.session_state or st.session_state.groq_client is None:
        if 'groq_api_key' in st.session_state and st.session_state.groq_api_key:
            try:
                # FIXED: Direct initialization with api_key
                st.session_state.groq_client = Groq(
                    api_key=st.session_state.groq_api_key
                )
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {str(e)}")
                return None
    return st.session_state.groq_client

# ============================================================================
# HELPER FUNCTIONS FROM PREVIOUS CELLS
# ============================================================================

# PDF Parser Functions (from Cell 4)
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

# Resume Analysis Functions (from Cell 5)
def calculate_ats_score(resume_data: Dict) -> Dict:
    """Calculate ATS compatibility score."""
    score = 0
    max_score = 100
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

def complete_resume_analysis(resume_data: Dict) -> Dict:
    """Complete resume analysis."""
    ats_result = calculate_ats_score(resume_data)
    gemini_analysis = analyze_resume_with_gemini(resume_data, st.session_state.gemini_model)
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

# Job Matching Functions (from Cell 6)
def extract_job_requirements(job_description: str) -> Dict:
    """Extract requirements from job description."""
    jd_lower = job_description.lower()
    
    skills_database = [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
        'kotlin', 'go', 'rust', 'typescript', 'sql', 'mysql', 'postgresql', 
        'mongodb', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'react', 
        'angular', 'vue', 'node.js', 'django', 'flask', 'machine learning',
        'data analysis', 'agile', 'scrum'
    ]
    
    required_skills = []
    for skill in skills_database:
        if skill in jd_lower:
            required_skills.append(skill.title())
    
    exp_pattern = r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of\s*)?(?:experience|exp)'
    exp_matches = re.findall(exp_pattern, jd_lower)
    required_experience = int(exp_matches[0]) if exp_matches else 0
    
    education_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'degree']
    required_education = any(keyword in jd_lower for keyword in education_keywords)
    
    return {
        'required_skills': required_skills,
        'required_experience': required_experience,
        'required_education': required_education,
        'total_words': len(job_description.split())
    }

def calculate_match_score(resume_data: Dict, job_requirements: Dict) -> Dict:
    """Calculate match score."""
    score = 0
    breakdown = {}
    
    # Skills (60 points)
    resume_skills = set([s.lower() for s in resume_data['skills']])
    required_skills = set([s.lower() for s in job_requirements['required_skills']])
    
    if required_skills:
        matching_skills = resume_skills.intersection(required_skills)
        missing_skills = required_skills - resume_skills
        skill_match_pct = len(matching_skills) / len(required_skills)
        skill_points = int(skill_match_pct * 60)
        score += skill_points
        breakdown['skills'] = {
            'score': skill_points,
            'max': 60,
            'matching': list(matching_skills),
            'missing': list(missing_skills),
            'match_percentage': round(skill_match_pct * 100, 1)
        }
    else:
        score += 30
        breakdown['skills'] = {'score': 30, 'max': 60, 'matching': [], 'missing': [], 'match_percentage': 50.0}
    
    # Experience (20 points)
    num_exp = len(resume_data['experience'])
    if num_exp >= job_requirements['required_experience']:
        exp_points = 20
    elif num_exp >= job_requirements['required_experience'] - 1:
        exp_points = 15
    else:
        exp_points = 10
    score += exp_points
    breakdown['experience'] = {'score': exp_points, 'max': 20, 'required': job_requirements['required_experience'], 'found': num_exp}
    
    # Education (10 points)
    has_edu = len(resume_data['education']) > 0
    edu_points = 10 if has_edu else 5
    score += edu_points
    breakdown['education'] = {'score': edu_points, 'max': 10, 'required': job_requirements['required_education'], 'found': has_edu}
    
    # Quality (10 points)
    quality = 0
    if resume_data['email'] != "Not found":
        quality += 3
    if resume_data['phone'] != "Not found":
        quality += 3
    if resume_data['total_words'] >= 300:
        quality += 4
    score += quality
    breakdown['quality'] = {'score': quality, 'max': 10}
    
    return {
        'total_score': score,
        'percentage': round(score, 1),
        'grade': 'Excellent Match' if score >= 80 else 'Good Match' if score >= 65 else 'Fair Match' if score >= 50 else 'Poor Match',
        'breakdown': breakdown
    }

def generate_job_match_report_with_gemini(resume_data: Dict, job_description: str, match_score: Dict, gemini_model) -> str:
    """Generate match report with Gemini."""
    prompt = f"""Career counselor analysis of job match.

JOB DESCRIPTION:
{job_description[:1500]}

CANDIDATE:
- Skills: {', '.join(resume_data['skills'][:20])}
- Experience: {len(resume_data['experience'])} roles
- Match Score: {match_score['total_score']}/100

Provide:
1. OVERALL FIT
2. KEY STRENGTHS (3-4 points)
3. GAPS (3-4 points)
4. INTERVIEW TIPS (4-5 points)
5. PREPARATION STEPS"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "Report unavailable."

def generate_tailored_cover_letter(resume_data: Dict, job_description: str) -> str:
    """Generate cover letter with Groq."""
    prompt = f"""Write a professional cover letter.

SKILLS: {', '.join(resume_data['skills'][:15])}
EXPERIENCE: {len(resume_data['experience'])} roles

JOB:
{job_description[:1000]}

Write 250-300 words covering:
1. Enthusiasm for role
2. Relevant skills/experiences
3. Understanding of needs
4. Call to action"""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return "Cover letter unavailable."

def complete_job_match_analysis(resume_data: Dict, job_description: str) -> Dict:
    """Complete job matching."""
    job_req = extract_job_requirements(job_description)
    match_score = calculate_match_score(resume_data, job_req)
    gemini_report = generate_job_match_report_with_gemini(resume_data, job_description, match_score, st.session_state.gemini_model)
    cover_letter = generate_tailored_cover_letter(resume_data, job_description)
    
    return {
        'match_score': match_score['total_score'],
        'match_percentage': match_score['percentage'],
        'match_grade': match_score['grade'],
        'breakdown': match_score['breakdown'],
        'gemini_report': gemini_report,
        'cover_letter': cover_letter,
        'missing_skills': match_score['breakdown']['skills']['missing'],
        'matching_skills': match_score['breakdown']['skills']['matching']
    }

# Interview Question Generation (from Cell 7)
def generate_technical_questions(resume_data: Dict, difficulty: str, num_questions: int) -> List[Dict]:
    """Generate technical questions."""
    skills_list = ', '.join(resume_data['skills'][:10])
    
    prompt = f"""Generate {num_questions} {difficulty} technical interview questions.

SKILLS: {skills_list}

Format:
Q1: [Question]
Expected Answer: [Answer]
Follow-up: [Follow-up]

Continue for all questions."""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            temperature=0.8
        )
        
        questions_text = response.choices[0].message.content
        questions = []
        blocks = questions_text.split('\n\n')
        
        for i, block in enumerate(blocks[:num_questions], 1):
            if block.strip():
                questions.append({
                    'id': i,
                    'type': 'Technical',
                    'difficulty': difficulty,
                    'question': block.strip(),
                    'category': 'Programming'
                })
        
        return questions
    except:
        return []

def generate_behavioral_questions(num_questions: int) -> List[Dict]:
    """Generate behavioral questions."""
    prompt = f"""Generate {num_questions} behavioral interview questions using STAR method.

Format:
Q1: [Question]
What to listen for: [Points]

Continue for all questions."""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content
        questions = []
        blocks = questions_text.split('\n\n')
        
        for i, block in enumerate(blocks[:num_questions], 1):
            if block.strip():
                questions.append({
                    'id': i,
                    'type': 'Behavioral',
                    'difficulty': 'Medium',
                    'question': block.strip(),
                    'category': 'Soft Skills'
                })
        
        return questions
    except:
        return []

def generate_situational_questions(resume_data: Dict, num_questions: int) -> List[Dict]:
    """Generate situational questions."""
    prompt = f"""Generate {num_questions} situational questions.

BACKGROUND:
- Experience: {len(resume_data['experience'])} roles
- Skills: {', '.join(resume_data['skills'][:8])}

Format:
Q1: [Scenario and question]
Evaluation: [Criteria]"""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            temperature=0.8
        )
        
        questions_text = response.choices[0].message.content
        questions = []
        blocks = questions_text.split('\n\n')
        
        for i, block in enumerate(blocks[:num_questions], 1):
            if block.strip():
                questions.append({
                    'id': i,
                    'type': 'Situational',
                    'difficulty': 'Medium',
                    'question': block.strip(),
                    'category': 'Problem Solving'
                })
        
        return questions
    except:
        return []

def generate_resume_based_questions(resume_data: Dict, num_questions: int) -> List[Dict]:
    """Generate resume-based questions."""
    skills = ', '.join(resume_data['skills'][:10])
    exp_summary = ""
    for exp in resume_data['experience'][:3]:
        exp_summary += f"- {exp['title']}\n"
    
    prompt = f"""Generate {num_questions} specific questions based on this resume.

SKILLS: {skills}
EXPERIENCE:
{exp_summary}

Format:
Q1: [Question about resume]
Why: [Reasoning]"""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content
        questions = []
        blocks = questions_text.split('\n\n')
        
        for i, block in enumerate(blocks[:num_questions], 1):
            if block.strip():
                questions.append({
                    'id': i,
                    'type': 'Resume-Based',
                    'difficulty': 'Medium',
                    'question': block.strip(),
                    'category': 'Experience'
                })
        
        return questions
    except:
        return []

def generate_complete_interview_set(resume_data: Dict, interview_type: str, difficulty: str) -> Dict:
    """Generate complete interview set."""
    all_questions = []
    
    if interview_type == "Technical":
        all_questions.extend(generate_technical_questions(resume_data, difficulty, 8))
        all_questions.extend(generate_resume_based_questions(resume_data, 3))
    elif interview_type == "Behavioral":
        all_questions.extend(generate_behavioral_questions(7))
        all_questions.extend(generate_situational_questions(resume_data, 4))
    elif interview_type == "Mixed":
        all_questions.extend(generate_technical_questions(resume_data, difficulty, 4))
        all_questions.extend(generate_behavioral_questions(3))
        all_questions.extend(generate_resume_based_questions(resume_data, 2))
    else:  # Full
        all_questions.extend(generate_technical_questions(resume_data, difficulty, 5))
        all_questions.extend(generate_behavioral_questions(4))
        all_questions.extend(generate_situational_questions(resume_data, 3))
        all_questions.extend(generate_resume_based_questions(resume_data, 3))
    
    return {
        'interview_type': interview_type,
        'difficulty': difficulty,
        'total_questions': len(all_questions),
        'questions': all_questions
    }

# Answer Evaluation (from Cell 8)
def evaluate_answer_with_groq(question: str, answer: str, question_type: str) -> Dict:
    """Evaluate interview answer."""
    prompt = f"""Evaluate this answer.

TYPE: {question_type}
QUESTION: {question}
ANSWER: {answer}

Score (0-10 each):
RELEVANCE: [score]/10
TECHNICAL_ACCURACY: [score]/10
COMMUNICATION: [score]/10
DEPTH: [score]/10
COMPLETENESS: [score]/10

TOTAL_SCORE: [sum]/50

STRENGTHS:
- [point]

WEAKNESSES:
- [point]

IMPROVED_ANSWER:
[better version]

FEEDBACK:
[2-3 sentences]"""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.3
        )
        
        eval_text = response.choices[0].message.content
        
        scores = {}
        try:
            relevance = re.search(r'RELEVANCE:\s*(\d+)', eval_text)
            technical = re.search(r'TECHNICAL_ACCURACY:\s*(\d+)', eval_text)
            communication = re.search(r'COMMUNICATION:\s*(\d+)', eval_text)
            depth = re.search(r'DEPTH:\s*(\d+)', eval_text)
            completeness = re.search(r'COMPLETENESS:\s*(\d+)', eval_text)
            
            scores = {
                'relevance': int(relevance.group(1)) if relevance else 7,
                'technical_accuracy': int(technical.group(1)) if technical else 7,
                'communication': int(communication.group(1)) if communication else 7,
                'depth': int(depth.group(1)) if depth else 6,
                'completeness': int(completeness.group(1)) if completeness else 7
            }
            
            scores['total'] = sum(scores.values())
            scores['percentage'] = round((scores['total'] / 50) * 100, 1)
        except:
            scores = {'relevance': 7, 'technical_accuracy': 7, 'communication': 7, 'depth': 6, 'completeness': 7, 'total': 34, 'percentage': 68.0}
        
        return {
            'success': True,
            'scores': scores,
            'full_evaluation': eval_text,
            'grade': get_grade(scores['percentage'])
        }
    except:
        return {'success': False, 'scores': {'total': 0, 'percentage': 0}, 'full_evaluation': "Unavailable", 'grade': 'N/A'}

def get_grade(percentage: float) -> str:
    """Convert percentage to grade."""
    if percentage >= 90:
        return 'A+ (Excellent)'
    elif percentage >= 80:
        return 'A (Very Good)'
    elif percentage >= 70:
        return 'B (Good)'
    elif percentage >= 60:
        return 'C (Fair)'
    elif percentage >= 50:
        return 'D (Below Average)'
    else:
        return 'F (Poor)'

def analyze_answer_sentiment(answer: str) -> Dict:
    """Analyze answer confidence."""
    answer_lower = answer.lower()
    confident_words = ['definitely', 'certainly', 'confident', 'sure', 'absolutely', 'experienced', 'skilled', 'proficient', 'expert']
    uncertain_words = ['maybe', 'perhaps', 'not sure', 'think', 'guess', 'probably', 'might', 'possibly']
    
    confident_count = sum(1 for word in confident_words if word in answer_lower)
    uncertain_count = sum(1 for word in uncertain_words if word in answer_lower)
    word_count = len(answer.split())
    
    if confident_count > uncertain_count:
        confidence = "High"
        confidence_score = min(90, 70 + (confident_count * 5))
    elif uncertain_count > confident_count:
        confidence = "Low"
        confidence_score = max(40, 60 - (uncertain_count * 5))
    else:
        confidence = "Moderate"
        confidence_score = 65
    
    return {
        'confidence_level': confidence,
        'confidence_score': confidence_score,
        'answer_length': word_count,
        'is_detailed': word_count >= 50
    }

def conduct_mock_interview(resume_data: Dict, questions: List[Dict], answers: List[str]) -> Dict:
    """Conduct complete interview evaluation."""
    interview_results = []
    total_score = 0
    total_possible = 0
    
    for i, (question_obj, answer) in enumerate(zip(questions, answers), 1):
        question = question_obj['question']
        question_type = question_obj['type']
        
        evaluation = evaluate_answer_with_groq(question, answer, question_type)
        sentiment = analyze_answer_sentiment(answer)
        
        result = {
            'question_number': i,
            'question': question,
            'question_type': question_type,
            'answer': answer,
            'evaluation': evaluation,
            'sentiment': sentiment,
            'scores': evaluation['scores'],
            'grade': evaluation['grade']
        }
        
        interview_results.append(result)
        
        if evaluation['success']:
            total_score += evaluation['scores']['total']
            total_possible += 50
    
    overall_pct = round((total_score / total_possible * 100), 1) if total_possible > 0 else 0
    
    strengths = []
    weaknesses = []
    
    for result in interview_results:
        if result['evaluation']['success']:
            if result['scores']['total'] >= 40:
                strengths.append(result['question_type'])
            elif result['scores']['total'] < 30:
                weaknesses.append(result['question_type'])
    
    return {
        'total_questions': len(questions),
        'total_answered': len(answers),
        'total_score': total_score,
        'total_possible': total_possible,
        'overall_percentage': overall_pct,
        'overall_grade': get_grade(overall_pct),
        'results': interview_results,
        'strengths': list(set(strengths)),
        'weaknesses': list(set(weaknesses)),
        'average_confidence': sum(r['sentiment']['confidence_score'] for r in interview_results) / len(interview_results) if interview_results else 0
    }

def generate_improvement_plan(interview_results: Dict) -> str:
    """Generate improvement plan."""
    weaknesses = ', '.join(interview_results['weaknesses']) if interview_results['weaknesses'] else "None"
    strengths = ', '.join(interview_results['strengths']) if interview_results['strengths'] else "Various"
    
    prompt = f"""Create improvement plan for candidate.

PERFORMANCE:
- Score: {interview_results['overall_percentage']}%
- Questions: {interview_results['total_questions']}
- Strengths: {strengths}
- Weaknesses: {weaknesses}
- Confidence: {interview_results['average_confidence']:.1f}%

Provide:
1. TOP 3 PRIORITIES
2. ACTION ITEMS (5-6 steps)
3. PRACTICE TIPS
4. 2-WEEK TIMELINE"""

    try:
        response = get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return "Plan unavailable."

# Session Management (from Cell 9)
SESSION_DATA = {
    'resume_data': None,
    'resume_analysis': None,
    'job_match_analysis': None,
    'interview_history': [],
    'current_interview': None,
    'user_profile': {
        'name': 'Candidate',
        'sessions_completed': 0,
        'total_questions_answered': 0,
        'average_score': 0
    }
}

def initialize_session():
    """Initialize session."""
    global SESSION_DATA
    SESSION_DATA = {
        'resume_data': None,
        'resume_analysis': None,
        'job_match_analysis': None,
        'interview_history': [],
        'current_interview': None,
        'user_profile': {
            'name': 'Candidate',
            'sessions_completed': 0,
            'total_questions_answered': 0,
            'average_score': 0
        }
    }

def save_interview_results(interview_results: Dict):
    """Save interview to history."""
    interview_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'interview_type': interview_results.get('interview_type', 'Mixed'),
        'total_questions': interview_results['total_questions'],
        'overall_score': interview_results['overall_percentage'],
        'grade': interview_results['overall_grade'],
        'results': interview_results['results']
    }
    
    SESSION_DATA['interview_history'].append(interview_record)
    SESSION_DATA['user_profile']['sessions_completed'] += 1
    SESSION_DATA['user_profile']['total_questions_answered'] += interview_results['total_questions']
    
    all_scores = [r['overall_score'] for r in SESSION_DATA['interview_history']]
    SESSION_DATA['user_profile']['average_score'] = round(sum(all_scores) / len(all_scores), 1)

def calculate_performance_trends() -> Dict:
    """Calculate performance trends."""
    if not SESSION_DATA['interview_history']:
        return {'trend': 'No data', 'improvement': 0, 'scores': []}
    
    scores = [r['overall_score'] for r in SESSION_DATA['interview_history']]
    
    if len(scores) < 2:
        return {'trend': 'Insufficient data', 'improvement': 0, 'scores': scores}
    
    first_score = scores[0]
    last_score = scores[-1]
    improvement = last_score - first_score
    
    if improvement > 10:
        trend = "Strong Improvement 📈"
    elif improvement > 0:
        trend = "Slight Improvement 📊"
    elif improvement == 0:
        trend = "Stable Performance ➡️"
    elif improvement > -10:
        trend = "Slight Decline 📉"
    else:
        trend = "Needs Attention ⚠️"
    
    return {
        'trend': trend,
        'improvement': round(improvement, 1),
        'scores': scores,
        'first_score': first_score,
        'last_score': last_score
    }

def analyze_question_type_performance() -> Dict:
    """Analyze by question type."""
    if not SESSION_DATA['interview_history']:
        return {}
    
    type_scores = {
        'Technical': [],
        'Behavioral': [],
        'Situational': [],
        'Resume-Based': []
    }
    
    for interview in SESSION_DATA['interview_history']:
        for result in interview['results']:
            q_type = result['question_type']
            if q_type in type_scores and result['evaluation']['success']:
                type_scores[q_type].append(result['scores']['percentage'])
    
    avg_scores = {}
    for q_type, scores in type_scores.items():
        if scores:
            avg_scores[q_type] = {
                'average': round(sum(scores) / len(scores), 1),
                'count': len(scores),
                'best': max(scores),
                'worst': min(scores)
            }
    
    return avg_scores

def create_performance_chart_data() -> Dict:
    """Prepare chart data."""
    if not SESSION_DATA['interview_history']:
        return {'has_data': False}
    
    sessions = list(range(1, len(SESSION_DATA['interview_history']) + 1))
    scores = [r['overall_score'] for r in SESSION_DATA['interview_history']]
    type_perf = analyze_question_type_performance()
    
    return {
        'has_data': True,
        'score_trend': {
            'sessions': sessions,
            'scores': scores
        },
        'type_performance': type_perf,
        'latest_score': scores[-1] if scores else 0,
        'average_score': SESSION_DATA['user_profile']['average_score']
    }

def export_session_data_to_json() -> str:
    """Export to JSON."""
    export_data = {
        'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_profile': SESSION_DATA['user_profile'],
        'interview_history': SESSION_DATA['interview_history'],
        'performance_trends': calculate_performance_trends(),
        'question_type_performance': analyze_question_type_performance()
    }
    return json.dumps(export_data, indent=2)

def generate_text_report(interview_results: Dict) -> str:
    """Generate text report."""
    report = f"""
{'=' * 70}
                    INTERVIEW PERFORMANCE REPORT
{'=' * 70}

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Interview Type: {interview_results.get('interview_type', 'Mixed')}

{'=' * 70}
OVERALL PERFORMANCE
{'=' * 70}

Total Questions: {interview_results['total_questions']}
Overall Score: {interview_results['total_score']}/{interview_results['total_possible']} ({interview_results['overall_percentage']}%)
Grade: {interview_results['overall_grade']}
Average Confidence: {interview_results['average_confidence']:.1f}%

{'=' * 70}
END OF REPORT
{'=' * 70}
"""
    return report

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

def get_motivational_message(score: float) -> str:
    """Get motivational message."""
    if score >= 90:
        return "Outstanding! You're interview-ready! 🎉"
    elif score >= 80:
        return "Excellent work! Keep it up! 💪"
    elif score >= 70:
        return "Good job! Keep practicing! 📚"
    elif score >= 60:
        return "You're making progress! 🎯"
    elif score >= 50:
        return "Keep going! 🚀"
    else:
        return "Don't give up! Practice makes perfect! 💡"

# ============================================================================
# STREAMLIT APP START
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="AI Interview Prep & Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Custom header styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Score card styles */
    .score-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .score-card.excellent {
        border-left-color: #10b981;
    }
    
    .score-card.good {
        border-left-color: #3b82f6;
    }
    
    .score-card.fair {
        border-left-color: #f59e0b;
    }
    
    .score-card.poor {
        border-left-color: #ef4444;
    }
    
    .score-value {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    .score-label {
        font-size: 1rem;
        color: #6b7280;
        margin: 0.5rem 0 0 0;
    }
    
    /* Info box styles */
    .info-box {
        background: #f3f4f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .info-box.success {
        background: #ecfdf5;
        border-left-color: #10b981;
    }
    
    .info-box.warning {
        background: #fffbeb;
        border-left-color: #f59e0b;
    }
    
    .info-box.error {
        background: #fef2f2;
        border-left-color: #ef4444;
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    
    /* Progress bar custom */
    .progress-bar {
        background: #e5e7eb;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: #f3f4f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Question card */
    .question-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .question-number {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.resume_data = None
    st.session_state.resume_analysis = None
    st.session_state.job_match_analysis = None
    st.session_state.interview_history = []
    st.session_state.current_interview = None
    st.session_state.api_keys_set = False
    st.session_state.groq_api_key = None
    st.session_state.gemini_api_key = None
    st.session_state.groq_client = None
    st.session_state.gemini_model = None

# Sidebar for API Keys
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    if not st.session_state.api_keys_set:
        st.info("Please enter your API keys to get started")
        
        groq_key = st.text_input("Groq API Key", type="password", help="Get from console.groq.com/keys")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Get from aistudio.google.com/app/apikey")
        
        if st.button("✅ Save API Keys"):
            if groq_key and gemini_key:
                try:
                    # Store keys in session state for later use
                    st.session_state.groq_api_key = groq_key
                    st.session_state.gemini_api_key = gemini_key
                    
                    # Initialize Gemini
                    genai.configure(api_key=gemini_key)
                    st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    
                    st.session_state.api_keys_set = True
                    st.success("✅ API keys saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.error("Please enter both API keys")
    else:
        st.success("✅ API Keys Configured")
        
        # Test Groq connection
        if st.button("🧪 Test Groq API"):
            try:
                client = get_groq_client()
                if client:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": "Say 'working' if you can hear me"}],
                        model="llama-3.3-70b-versatile",
                        max_tokens=10
                    )
                    st.success(f"✅ Groq API working! Response: {response.choices[0].message.content}")
                else:
                    st.error("❌ Groq client is None")
            except Exception as e:
                st.error(f"❌ Groq test failed: {str(e)}")
        
        if st.button("🔄 Reset API Keys"):
            st.session_state.api_keys_set = False
            st.session_state.groq_api_key = None
            st.session_state.gemini_api_key = None
            st.session_state.groq_client = None
            st.session_state.gemini_model = None
            st.rerun()
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Go to:",
        ["🏠 Home", "📄 Resume Analysis", "🎯 Job Matcher", "🎤 Mock Interview", "📊 Dashboard"],
        label_visibility="collapsed",
        key="navigation_radio"  # Add this unique key
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if st.session_state.resume_data:
        st.metric("Skills Found", len(st.session_state.resume_data.get('skills', [])))
    if st.session_state.interview_history:
        st.metric("Interviews Done", len(st.session_state.interview_history))
        avg_score = sum(i['overall_score'] for i in st.session_state.interview_history) / len(st.session_state.interview_history)
        st.metric("Average Score", f"{avg_score:.1f}%")

    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Go to:",
        ["🏠 Home", "📄 Resume Analysis", "🎯 Job Matcher", "🎤 Mock Interview", "📊 Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if st.session_state.resume_data:
        st.metric("Skills Found", len(st.session_state.resume_data.get('skills', [])))
    if st.session_state.interview_history:
        st.metric("Interviews Done", len(st.session_state.interview_history))
        avg_score = sum(i['overall_score'] for i in st.session_state.interview_history) / len(st.session_state.interview_history)
        st.metric("Average Score", f"{avg_score:.1f}%")

# Main content area
if not st.session_state.api_keys_set:
    # Welcome screen when APIs not configured
    st.markdown("""
        <div class="main-header">
            <h1>🎯 AI Interview Prep & Resume Analyzer</h1>
            <p>Your AI-powered career preparation assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 👈 Get Started")
    st.info("Enter your API keys in the sidebar to begin!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### 🔑 Get Groq API Key
            1. Visit [console.groq.com/keys](https://console.groq.com/keys)
            2. Sign up/Login with Google
            3. Click 'Create API Key'
            4. Copy and paste in sidebar
        """)
    
    with col2:
        st.markdown("""
            ### 🔑 Get Gemini API Key
            1. Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
            2. Sign in with Google
            3. Click 'Create API Key'
            4. Copy and paste in sidebar
        """)
    
    st.markdown("---")
    st.markdown("## ✨ Features")
    
    feat1, feat2, feat3 = st.columns(3)
    
    with feat1:
        st.markdown("""
            <div class="info-box success">
                <h3>📄 Resume Analysis</h3>
                <p>Get ATS score, skill analysis, and improvement suggestions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat2:
        st.markdown("""
            <div class="info-box success">
                <h3>🎯 Job Matching</h3>
                <p>Compare your resume with job descriptions and get match scores</p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat3:
        st.markdown("""
            <div class="info-box success">
                <h3>🎤 Mock Interviews</h3>
                <p>Practice with AI-generated questions and get real-time feedback</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # Main application pages
    
    if page == "🏠 Home":
        st.markdown("""
            <div class="main-header">
                <h1>🎯 AI Interview Prep & Resume Analyzer</h1>
                <p>Your AI-powered career preparation assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Upload Resume", use_container_width=True):
                st.session_state.page = "📄 Resume Analysis"
                st.rerun()
        
        with col2:
            if st.button("🎯 Match Job", use_container_width=True):
                st.session_state.page = "🎯 Job Matcher"
                st.rerun()
        
        with col3:
            if st.button("🎤 Start Interview", use_container_width=True):
                st.session_state.page = "🎤 Mock Interview"
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.resume_data:
            st.markdown("## 📊 Your Profile Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.resume_data.get('skills', []))}</div>
                        <div class="metric-label">Skills Identified</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                exp_count = len(st.session_state.resume_data.get('experience', []))
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{exp_count}</div>
                        <div class="metric-label">Experience Entries</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if st.session_state.resume_analysis:
                    ats_score = st.session_state.resume_analysis.get('ats_score', 0)
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{ats_score}</div>
                            <div class="metric-label">ATS Score</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload your resume to get started!")
    
    elif page == "📄 Resume Analysis":
        st.markdown("""
            <div class="main-header">
                <h1>📄 Resume Analysis</h1>
                <p>Upload your resume for comprehensive AI analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
        
        if uploaded_file:
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Analyzing your resume..."):
                    # Extract text from PDF
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        # Structure resume data
                        resume_data = structure_resume_data(text)
                        st.session_state.resume_data = resume_data
                        
                        # Perform complete analysis
                        analysis = complete_resume_analysis(resume_data)
                        st.session_state.resume_analysis = analysis
                        
                        st.success("✅ Resume analysis complete!")
                        st.rerun()
        
        if st.session_state.resume_analysis:
            st.markdown("---")
            
            # Display ATS Score
            st.markdown("## 📊 ATS Compatibility Score")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                ats_score = st.session_state.resume_analysis['ats_score']
                st.markdown(f"""
                    <div class="score-card {'excellent' if ats_score >= 80 else 'good' if ats_score >= 70 else 'fair'}">
                        <p class="score-value">{ats_score}</p>
                        <p class="score-label">ATS Score / 100</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                grade = st.session_state.resume_analysis['ats_grade']
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{grade}</div>
                        <div class="metric-label">Grade</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                badge = format_score_badge(ats_score)
                st.markdown(f"""
                    <div class="info-box success">
                        <h3 style="margin:0;">{badge}</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            # Score breakdown
            st.markdown("### 📋 Score Breakdown")
            for feedback in st.session_state.resume_analysis['ats_feedback']:
                if '✅' in feedback:
                    st.markdown(f'<div class="info-box success">{feedback}</div>', unsafe_allow_html=True)
                elif '⚠️' in feedback:
                    st.markdown(f'<div class="info-box warning">{feedback}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-box error">{feedback}</div>', unsafe_allow_html=True)
            
            # Gemini AI Analysis
            st.markdown("---")
            st.markdown("## 🤖 AI-Powered Analysis")
            
            with st.expander("📝 Detailed Analysis", expanded=True):
                st.markdown(st.session_state.resume_analysis['gemini_analysis'])
            
            # Improvement Suggestions
            st.markdown("---")
            st.markdown("## 💡 Improvement Suggestions")
            
            for i, suggestion in enumerate(st.session_state.resume_analysis['improvement_suggestions'], 1):
                st.markdown(f"{i}. {suggestion}")
            
            # Skills Display
            st.markdown("---")
            st.markdown("## 🎯 Identified Skills")
            
            skills = st.session_state.resume_data['skills']
            skills_html = '<div style="margin: 1rem 0;">'
            for skill in skills:
                skills_html += f'<span class="badge badge-info">{skill}</span>'
            skills_html += '</div>'
            st.markdown(skills_html, unsafe_allow_html=True)

    elif page == "🎯 Job Matcher":
        st.markdown("""
            <div class="main-header">
                <h1>🎯 Job Description Matcher</h1>
                <p>Compare your resume with job descriptions</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.resume_data:
            st.warning("⚠️ Please upload and analyze your resume first!")
            if st.button("📄 Go to Resume Analysis"):
                st.session_state.page = "📄 Resume Analysis"
                st.rerun()
        else:
            st.markdown("## 📋 Paste Job Description")
            
            job_description = st.text_area(
                "Job Description",
                height=300,
                placeholder="Paste the complete job description here...",
                help="Include requirements, qualifications, and responsibilities"
            )
            
            if st.button("🔍 Analyze Match", type="primary", disabled=not job_description):
                with st.spinner("Analyzing job match..."):
                    try:
                        # Perform job match analysis
                        match_analysis = complete_job_match_analysis(
                            st.session_state.resume_data,
                            job_description
                        )
                        st.session_state.job_match_analysis = match_analysis
                        st.success("✅ Job match analysis complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
            
            if st.session_state.job_match_analysis:
                st.markdown("---")
                
                # Match Score Display
                st.markdown("## 📊 Match Score")
                
                match_score = st.session_state.job_match_analysis['match_percentage']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                        <div class="score-card {'excellent' if match_score >= 80 else 'good' if match_score >= 65 else 'fair'}">
                            <p class="score-value">{match_score:.1f}%</p>
                            <p class="score-label">Match Score</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    grade = st.session_state.job_match_analysis['match_grade']
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="font-size: 1.5rem;">{grade}</div>
                            <div class="metric-label">Match Grade</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Score Breakdown
                st.markdown("---")
                st.markdown("## 📈 Detailed Breakdown")
                
                breakdown = st.session_state.job_match_analysis['breakdown']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    skills_score = breakdown['skills']['score']
                    skills_max = breakdown['skills']['max']
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{skills_score}/{skills_max}</div>
                            <div class="metric-label">Skills Match</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    exp_score = breakdown['experience']['score']
                    exp_max = breakdown['experience']['max']
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{exp_score}/{exp_max}</div>
                            <div class="metric-label">Experience</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    edu_score = breakdown['education']['score']
                    edu_max = breakdown['education']['max']
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{edu_score}/{edu_max}</div>
                            <div class="metric-label">Education</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    quality_score = breakdown['quality']['score']
                    quality_max = breakdown['quality']['max']
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{quality_score}/{quality_max}</div>
                            <div class="metric-label">Profile Quality</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Skills Analysis
                st.markdown("---")
                st.markdown("## 🎯 Skills Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Matching Skills")
                    matching_skills = st.session_state.job_match_analysis['matching_skills']
                    if matching_skills:
                        skills_html = '<div style="margin: 1rem 0;">'
                        for skill in matching_skills:
                            skills_html += f'<span class="badge badge-success">{skill}</span>'
                        skills_html += '</div>'
                        st.markdown(skills_html, unsafe_allow_html=True)
                    else:
                        st.info("No matching skills found")
                
                with col2:
                    st.markdown("### ❌ Missing Skills")
                    missing_skills = st.session_state.job_match_analysis['missing_skills']
                    if missing_skills:
                        skills_html = '<div style="margin: 1rem 0;">'
                        for skill in missing_skills[:10]:  # Limit to 10
                            skills_html += f'<span class="badge badge-danger">{skill}</span>'
                        skills_html += '</div>'
                        st.markdown(skills_html, unsafe_allow_html=True)
                    else:
                        st.success("All required skills present!")
                
                # AI Match Report
                st.markdown("---")
                st.markdown("## 🤖 AI Match Analysis")
                
                with st.expander("📝 Detailed Match Report", expanded=True):
                    st.markdown(st.session_state.job_match_analysis['gemini_report'])
                
                # Cover Letter
                st.markdown("---")
                st.markdown("## ✍️ AI-Generated Cover Letter")
                
                cover_letter = st.session_state.job_match_analysis['cover_letter']
                st.text_area("Your Cover Letter", cover_letter, height=400)
                
                # Download button
                st.download_button(
                    label="📥 Download Cover Letter",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain"
                )
    
    elif page == "🎤 Mock Interview":
        st.markdown("""
            <div class="main-header">
                <h1>🎤 Mock Interview Practice</h1>
                <p>Practice with AI-generated interview questions</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.resume_data:
            st.warning("⚠️ Please upload and analyze your resume first!")
            if st.button("📄 Go to Resume Analysis"):
                st.session_state.page = "📄 Resume Analysis"
                st.rerun()
        else:
            # Interview Configuration
            st.markdown("## ⚙️ Interview Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                interview_type = st.selectbox(
                    "Interview Type",
                    ["Technical", "Behavioral", "Mixed", "Full"],
                    help="Choose the type of interview you want to practice"
                )
            
            with col2:
                difficulty = st.selectbox(
                    "Difficulty Level",
                    ["Easy", "Medium", "Hard"],
                    index=1
                )
            
            if st.button("🎯 Generate Interview Questions", type="primary"):
                with st.spinner("Generating personalized questions..."):
                    try:
                        # Generate questions
                        interview_set = generate_complete_interview_set(
                            st.session_state.resume_data,
                            interview_type,
                            difficulty
                        )
                        st.session_state.current_interview = {
                            'questions': interview_set['questions'],
                            'answers': [],
                            'current_question': 0,
                            'interview_type': interview_type,
                            'difficulty': difficulty
                        }
                        st.success(f"✅ Generated {len(interview_set['questions'])} questions!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Question generation failed: {str(e)}")
            
# Interview Session
            if st.session_state.current_interview:
                st.markdown("---")
                
                questions = st.session_state.current_interview['questions']
                current_q = st.session_state.current_interview['current_question']
                
                # Check if questions exist
                if not questions or len(questions) == 0:
                    st.error("❌ No questions were generated. Please try again.")
                    if st.button("🔄 Generate New Questions"):
                        st.session_state.current_interview = None
                        st.rerun()
                else:
                    # Progress bar
                    progress = (current_q / len(questions)) * 100
                    st.progress(current_q / len(questions))
                    st.markdown(f"**Progress:** Question {current_q + 1} of {len(questions)} ({progress:.0f}%)")
                    
                    if current_q < len(questions):
                        # Display current question
                        question_obj = questions[current_q]
                        
                        st.markdown(f"""
                            <div class="question-card">
                                <span class="question-number">Question {current_q + 1}</span>
                                <span class="badge badge-info" style="margin-left: 0.5rem;">{question_obj['type']}</span>
                                <span class="badge badge-warning" style="margin-left: 0.5rem;">{question_obj['difficulty']}</span>
                                <p style="margin-top: 1rem; font-size: 1.1rem; color: #1f2937;">{question_obj['question']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Answer input
                        answer = st.text_area(
                            "Your Answer",
                            height=200,
                            placeholder="Type your answer here...",
                            key=f"answer_{current_q}"
                        )
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            if st.button("⏭️ Skip Question"):
                                st.session_state.current_interview['answers'].append("")
                                st.session_state.current_interview['current_question'] += 1
                                st.rerun()
                        
                        with col2:
                            if st.button("✅ Submit Answer", type="primary", disabled=not answer):
                                st.session_state.current_interview['answers'].append(answer)
                                st.session_state.current_interview['current_question'] += 1
                                st.rerun()
                    
                    else:
                        # Interview completed - show evaluation
                        st.markdown("---")
                        st.success("🎉 Interview Complete! Evaluating your answers...")
                        
                        if st.button("📊 View Results", type="primary"):
                            with st.spinner("Evaluating your performance..."):
                                try:
                                    # Evaluate all answers
                                    results = conduct_mock_interview(
                                        st.session_state.resume_data,
                                        questions,
                                        st.session_state.current_interview['answers']
                                    )
                                    
                                    # Save to history
                                    results['interview_type'] = st.session_state.current_interview['interview_type']
                                    save_interview_results(results)
                                    
                                    # Clear current interview
                                    st.session_state.current_interview = None
                                    st.session_state.interview_results = results
                                    
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Evaluation failed: {str(e)}")
            
            # Show results if available
            if 'interview_results' in st.session_state and st.session_state.interview_results:
                st.markdown("---")
                st.markdown("## 📊 Interview Results")
                
                results = st.session_state.interview_results
                
                # Overall stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div class="score-card {'excellent' if results['overall_percentage'] >= 80 else 'good' if results['overall_percentage'] >= 70 else 'fair'}">
                            <p class="score-value">{results['overall_percentage']:.1f}%</p>
                            <p class="score-label">Overall Score</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{results['total_questions']}</div>
                            <div class="metric-label">Total Questions</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(results['strengths'])}</div>
                            <div class="metric-label">Strong Areas</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{results['average_confidence']:.0f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Motivation message
                msg = get_motivational_message(results['overall_percentage'])
                st.markdown(f'<div class="info-box success"><h3>{msg}</h3></div>', unsafe_allow_html=True)
                
                # Question-by-question results
                st.markdown("---")
                st.markdown("## 📝 Detailed Results")
                
                for i, result in enumerate(results['results'], 1):
                    with st.expander(f"Question {i} - Score: {result['scores']['percentage']:.1f}% ({result['grade']})"):
                        st.markdown(f"**Question ({result['question_type']}):**")
                        st.info(result['question'])
                        
                        st.markdown("**Your Answer:**")
                        st.text_area("", result['answer'], height=100, disabled=True, key=f"result_answer_{i}")
                        
                        if result['evaluation']['success']:
                            st.markdown("**Score Breakdown:**")
                            scores = result['scores']
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Relevance", f"{scores['relevance']}/10")
                            with col2:
                                st.metric("Accuracy", f"{scores['technical_accuracy']}/10")
                            with col3:
                                st.metric("Communication", f"{scores['communication']}/10")
                            with col4:
                                st.metric("Depth", f"{scores['depth']}/10")
                            with col5:
                                st.metric("Completeness", f"{scores['completeness']}/10")
                            
                            st.markdown("**AI Feedback:**")
                            st.markdown(result['evaluation']['full_evaluation'])
                
                # Improvement Plan
                st.markdown("---")
                st.markdown("## 📈 Personalized Improvement Plan")
                
                improvement_plan = generate_improvement_plan(results)
                st.markdown(improvement_plan)
                
                # Download report
                st.markdown("---")
                report = generate_text_report(results)
                st.download_button(
                    label="📥 Download Full Report",
                    data=report,
                    file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                if st.button("🔄 Start New Interview"):
                    st.session_state.interview_results = None
                    st.rerun()

    elif page == "📊 Dashboard":
        st.markdown("""
            <div class="main-header">
                <h1>📊 Performance Dashboard</h1>
                <p>Track your interview preparation progress</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not SESSION_DATA['interview_history']:
            st.info("🎤 Complete some mock interviews to see your performance dashboard!")
            
            if st.button("🎯 Start Your First Interview"):
                st.session_state.page = "🎤 Mock Interview"
                st.rerun()
        else:
            # Overall Statistics
            st.markdown("## 📈 Overall Statistics")
            
            total_interviews = len(SESSION_DATA['interview_history'])
            total_questions = SESSION_DATA['user_profile']['total_questions_answered']
            avg_score = SESSION_DATA['user_profile']['average_score']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_interviews}</div>
                        <div class="metric-label">Interviews Completed</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_questions}</div>
                        <div class="metric-label">Questions Answered</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="score-card {'excellent' if avg_score >= 80 else 'good' if avg_score >= 70 else 'fair'}">
                        <p class="score-value">{avg_score:.1f}%</p>
                        <p class="score-label">Average Score</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Performance Trend Chart
            st.markdown("---")
            st.markdown("## 📈 Score Trend Over Time")
            
            chart_data = create_performance_chart_data()
            
            if chart_data['has_data']:
                # Line chart for score trend
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=chart_data['score_trend']['sessions'],
                    y=chart_data['score_trend']['scores'],
                    mode='lines+markers',
                    name='Score',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, color='#764ba2')
                ))
                
                fig.update_layout(
                    title="Interview Score Progression",
                    xaxis_title="Interview Session",
                    yaxis_title="Score (%)",
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance trend analysis
                trends = calculate_performance_trends()
                
                if trends['improvement'] > 0:
                    st.success(f"📈 {trends['trend']} - You've improved by {trends['improvement']:.1f}% since your first interview!")
                elif trends['improvement'] == 0:
                    st.info(f"➡️ {trends['trend']} - Maintain consistency and keep practicing!")
                else:
                    st.warning(f"📉 {trends['trend']} - Focus on your weak areas to improve!")
            
            # Question Type Performance
            st.markdown("---")
            st.markdown("## 🎯 Performance by Question Type")
            
            type_performance = analyze_question_type_performance()
            
            if type_performance:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    types = list(type_performance.keys())
                    averages = [type_performance[t]['average'] for t in types]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=types,
                            y=averages,
                            marker_color=['#667eea', '#764ba2', '#f59e0b', '#10b981']
                        )
                    ])
                    
                    fig.update_layout(
                        title="Average Score by Question Type",
                        xaxis_title="Question Type",
                        yaxis_title="Average Score (%)",
                        height=350,
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=averages,
                        theta=types,
                        fill='toself',
                        fillcolor='rgba(102, 126, 234, 0.3)',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        title="Skill Radar Chart",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                st.markdown("### 📊 Detailed Breakdown")
                
                for q_type, stats in type_performance.items():
                    with st.expander(f"{q_type} Questions"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Average", f"{stats['average']:.1f}%")
                        with col2:
                            st.metric("Questions", stats['count'])
                        with col3:
                            st.metric("Best", f"{stats['best']:.1f}%")
                        with col4:
                            st.metric("Worst", f"{stats['worst']:.1f}%")
            
            # Interview History
            st.markdown("---")
            st.markdown("## 📚 Interview History")
            
            history_data = []
            for i, interview in enumerate(reversed(SESSION_DATA['interview_history']), 1):
                history_data.append({
                    'Session': len(SESSION_DATA['interview_history']) - i + 1,
                    'Date': interview['timestamp'],
                    'Type': interview['interview_type'],
                    'Questions': interview['total_questions'],
                    'Score': f"{interview['overall_score']:.1f}%",
                    'Grade': interview['grade']
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export Options
            st.markdown("---")
            st.markdown("## 💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export JSON
                json_data = export_session_data_to_json()
                st.download_button(
                    label="📥 Download Session Data (JSON)",
                    data=json_data,
                    file_name=f"interview_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Clear history option
                if st.button("🗑️ Clear All History", type="secondary"):
                    if st.checkbox("⚠️ Are you sure? This cannot be undone!"):
                        initialize_session()
                        st.success("✅ History cleared!")
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p>Made with ❤️ using Streamlit, Groq, and Gemini AI</p>
        <p style="font-size: 0.9rem;">🎯 AI Interview Prep & Resume Analyzer v1.0</p>
    </div>
""", unsafe_allow_html=True)

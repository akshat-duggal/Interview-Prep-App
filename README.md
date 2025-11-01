# 🎯 AI Interview Prep & Resume Analyzer

An AI-powered interview preparation and resume analysis tool built with Streamlit, Groq, and Gemini AI.

## ✨ Features

- **📄 Resume Analysis**: Upload PDF resumes and get comprehensive ATS compatibility scores
- **🎯 Job Matching**: Compare your resume with job descriptions and get match scores
- **🎤 Mock Interviews**: Practice with AI-generated interview questions
- **📊 Performance Dashboard**: Track your progress with detailed analytics
- **🤖 AI-Powered**: Uses Groq (fast inference) and Gemini (multimodal analysis)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get it here](https://console.groq.com/keys))
- Gemini API Key ([Get it here](https://aistudio.google.com/app/apikey))

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd interview_prep_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

1. Push this code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!
5. Add your API keys in Streamlit Cloud secrets:
   - Go to App Settings → Secrets
   - Add your Groq and Gemini API keys

## 📖 How to Use

### 1. Resume Analysis
- Upload your PDF resume
- Get instant ATS compatibility score
- View AI-powered suggestions for improvement
- See identified skills and experience

### 2. Job Matching
- Paste any job description
- Get match score (0-100%)
- See which skills match and which are missing
- Generate tailored cover letter automatically

### 3. Mock Interview
- Choose interview type (Technical/Behavioral/Mixed/Full)
- Select difficulty (Easy/Medium/Hard)
- Answer AI-generated questions
- Get detailed feedback on each answer
- Receive personalized improvement plan

### 4. Performance Dashboard
- Track your interview scores over time
- Analyze performance by question type
- View strengths and weaknesses
- Export your data

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Groq (Llama 3.3 70B) - Fast inference for real-time responses
  - Gemini 2.5 Flash - Multimodal analysis and vision capabilities
- **PDF Processing**: PyPDF2, pdfplumber
- **Visualization**: Plotly
- **Data**: Pandas

## 📊 Project Structure
```
interview_prep_app/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🎓 Use Cases

- **Students**: Prepare for campus placements
- **Job Seekers**: Practice interviews and optimize resumes
- **Career Switchers**: Get feedback on new domain preparation
- **Professionals**: Keep interview skills sharp

## 🔒 Privacy & Security

- All processing happens in real-time
- No data is stored permanently on servers
- API keys are securely managed
- Session data is temporary

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

MIT License - feel free to use this project for your own purposes!

## 👨‍💻 Author

Built as a Generative AI course project

## 🙏 Acknowledgments

- Anthropic for Groq API
- Google for Gemini AI
- Streamlit for the amazing framework

---

**⭐ If you find this helpful, please star the repository!**

# Indian LawBot -AI Legal Assistant

Indian LawBot is an AI-powered legal assistant that:
- 🗣 Accepts *text and voice* questions
- 📘 Understands and answers questions based on *Indian legal documents*
- 🌐 Translates and simplifies legal content for the general public
- ⚖ Suggests *which court* to approach based on your case

---
## 🚀 Why This Project Matters

- 🧠 *AI meets LegalTech* – Understand Indian laws without a law degree  
- 💬 *Multilingual + Voice Enabled* – Breaks language and literacy barriers  
- 🧾 *Built on Real Lawbooks* – Answers come from IPC, CPC, Constitution, POSH, etc.  
- 💼 *End-to-End Legal Flow* – From question to advice to court mapping

---
## 🚀 Features
- 💬 Ask legal queries (in any language)
- 🎙 Voice-to-text and text-to-speech enabled
- 📄 Upload your legal documents (PDF/TXT)
- 🧠 Summarizes and answers from legal texts
- 🏛 Court recommender system
- 👩‍⚖ Special focus on *Women & Child Laws*

---
## 🧠 Backend & Training

- Built using `LangChain`, `ChromaDB`, `HuggingFace Embeddings`
- On-the-fly vector store generation from 20+ Indian law PDFs
- Powered by [Groq's Llama3-8B](https://console.groq.com/) API for ultra-fast reasoning

---

## 📁 Folder Structure


lawbot/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Contains 20+ Indian law PDFs
│   ├── Constitution of India.pdf
│   ├── Bharatiya Nyaya Sanhita.pdf
│   └── ...
└── .streamlit/               # Streamlit secrets (not uploaded to GitHub)
    └── secrets.toml
```

---

## 🔐 API Keys

This app uses **Groq's LLM API**, stored securely using Streamlit secrets:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_groq_api_key"
```

---

## 🟢 Live Demo

> ✅ [Click to Try It on Streamlit Cloud](https://lawbot-india.streamlit.app)  
*(Link placeholder – update it after deployment)*

---

## 📦 Run Locally

```bash
git clone https://github.com/yourusername/lawbot.git
cd lawbot
pip install -r requirements.txt
streamlit run app.py
```

Make sure to add your law PDFs to the `data/` folder.

---

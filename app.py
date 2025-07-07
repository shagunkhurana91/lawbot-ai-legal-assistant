import os
import zipfile
import fitz
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from gtts import gTTS
import base64

torch.set_default_device("cpu")

def play_audio_from_text(text, lang_code="en"):
    tts = gTTS(text=text, lang=lang_code)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""
        <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

st.set_page_config(page_title="Indian LawBot", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "faiss_index"
    zip_path = "faiss_index.zip"

    if not os.path.exists(persist_dir) and os.path.exists(zip_path):
        os.makedirs(persist_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith("index.faiss") or file.endswith("index.pkl"):
                    zip_ref.extract(file, persist_dir)

    index_faiss = os.path.join(persist_dir, "index.faiss")
    index_pkl = os.path.join(persist_dir, "index.pkl")

    if not os.path.exists(index_faiss) or not os.path.exists(index_pkl):
        st.error("❌ FAISS files not found after extraction. Ensure index.faiss and index.pkl are in the zip.")
        st.stop()

    try:
        return FAISS.load_local(persist_dir, embeddings)
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        st.stop()



st.sidebar.title("⚖️ Indian LawBot")
st.sidebar.markdown("""
### 🛠 Quick Actions
- ❓ Ask legal questions
- 📄 Upload documents
- ⚖️ Find the right court

Ʝ *This assistant simplifies legal language and offers guidance before consulting a lawyer.*
""")

tab1, tab2, tab3 = st.tabs(["❓ Legal Q&A", "📄 Document Help", "⚖️ Court Finder"])

with tab1:
    st.header("❓ Ask Your Legal Question")
    st.subheader("💬 Enter or 🎙️ Record Your Legal Question")

    query_input = st.text_input("📝 Type your legal question (any language):", key="qa_query")
    st.markdown("##### 🎤 Or upload your voice question (WAV format)")
    audio_file = st.file_uploader("Upload audio file", type=["wav"])

    voice_query = None

    if audio_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                temp_file_path = tmp.name

            sound = AudioSegment.from_file(temp_file_path)
            pcm_wav_path = temp_file_path.replace(".wav", "_converted.wav")
            sound.export(pcm_wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

            recognizer = sr.Recognizer()
            with sr.AudioFile(pcm_wav_path) as source:
                audio_data = recognizer.record(source)

            voice_query = recognizer.recognize_google(audio_data)
            st.success(f"🗣️ Transcribed voice: {voice_query}")
        except Exception as e:
            st.error(f"❌ Audio processing failed: {e}")

    final_query = query_input.strip() if query_input.strip() else (voice_query or "").strip()

    if final_query:
        st.markdown(f"**Query Used:** `{final_query}`")
        detected_lang = detect(final_query)
        translated_query = GoogleTranslator(source="auto", target="en").translate(final_query)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), return_source_documents=True)
        result = qa_chain(translated_query)
        legal_answer_en = result['result']
        legal_answer_translated = GoogleTranslator(source='en', target=detected_lang).translate(legal_answer_en)

        prompt = f"""
You're a legal expert. Explain this legal content simply like you're helping someone unfamiliar with legal jargon:
{legal_answer_en}
"""
        simplified = llm.invoke(prompt).content.strip()
        simplified_translated = GoogleTranslator(source='en', target=detected_lang).translate(simplified)

        view_mode = st.radio("Choose view mode:", ["🔍 Legal Answer", "✨ Simplified Explanation"], key="qa_view_mode")

        if view_mode == "🔍 Legal Answer":
            st.markdown("### ⚖️ Legal Answer")
            st.success(legal_answer_translated)
            if st.button("🔈 Play Legal Answer Audio"):
                try:
                    lang_code = detect(legal_answer_translated)
                    play_audio_from_text(legal_answer_translated, lang_code)
                except Exception as e:
                    st.warning(f"Audio playback failed: {e}")

        if view_mode == "✨ Simplified Explanation":
            st.markdown("### ✨ Simplified Explanation")
            st.info(simplified_translated)
            if st.button("🔈 Play Simplified Explanation Audio"):
                try:
                    lang_code = detect(simplified_translated)
                    play_audio_from_text(simplified_translated, lang_code)
                except Exception as e:
                    st.warning(f"Audio playback failed: {e}")
# ========== TAB 2 ==========
with tab2:
    st.header("📤 Upload and Analyze Legal Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT legal document", type=['pdf', 'txt'], key="doc_upload")

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        raw_text = ""
        if ext == "pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            raw_text = "\n".join([page.get_text() for page in doc])
        elif ext == "txt":
            raw_text = uploaded_file.read().decode("utf-8")

        if not raw_text.strip():
            st.warning("⚠ No text found in the uploaded document.")
        else:
            detected_lang = detect(raw_text[:500])
            action = st.radio("Choose action:", ["Ask Questions", "Summarize Document"], key="doc_action")

            if action == "Ask Questions":
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_text(raw_text)
                docs = [Document(page_content=chunk) for chunk in chunks]
                embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectordb = FAISS.from_documents(docs, embedding=embedder)
                local_qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_kwargs={"k": 5}), return_source_documents=True)

                user_q = st.text_input("Ask from the document:", key="doc_user_q")
                if user_q:
                    translated_q = GoogleTranslator(source="auto", target="en").translate(user_q)
                    result = local_qa(translated_q)
                    ans_en = result["result"]
                    ans_trans = GoogleTranslator(source="en", target=detected_lang).translate(ans_en)
                    st.success(ans_trans)

            elif action == "Summarize Document":
                short_prompt = f"""
You're a legal assistant. Give a simple, one-paragraph summary of this legal content for a general user.
Avoid legal jargon. Focus on what the document is about, the issue it covers, and what it means for the user.
CONTENT:
{raw_text[:3000]}
"""
                try:
                    response = llm.invoke(short_prompt).content.strip()
                    translated = GoogleTranslator(source="en", target=detected_lang).translate(response)
                    st.markdown("### 📄 Document Summary")
                    st.info(translated.strip())
                except Exception as e:
                    st.error(f"❌ Error during summarization: {e}")

# ========== TAB 3 ==========
with tab3:
    st.header("⚖️ Find the Right Court for Your Case")

    with st.form("court_finder_form"):
        case_type = st.selectbox("⚖ Type of Case", ["Criminal", "Civil", "Family", "Consumer", "Labour", "Property", "Appeal", "Other"])
        amount_involved = st.text_input("💰 Claim Amount (for Civil/Consumer only)", placeholder="e.g., 50000")
        special_case = st.selectbox("❗ Special Case Type (if applicable)", ["No", "Public Interest Litigation (PIL)", "Writ Petition", "Appeal in High Court"])
        submitted = st.form_submit_button("🔍 Recommend Court")

    def recommend_court(case_type, amount_involved, special_case):
        if special_case == "Public Interest Litigation (PIL)":
            return "High Court or Supreme Court – File under Article 226 or 32."
        elif special_case == "Writ Petition":
            return "High Court – Filed under Article 226."
        elif special_case == "Appeal in High Court":
            return "High Court – Appeals from District/Tribunal judgments."

        try:
            amount = int(amount_involved)
        except:
            amount = None

        if case_type == "Criminal":
            return "Approach the Judicial Magistrate or Sessions Court based on the IPC section."
        elif case_type == "Civil":
            if amount:
                if amount <= 100000:
                    return "Junior Civil Judge Court (Small Causes Court)."
                elif amount <= 1000000:
                    return "Senior Civil Judge Court (Subordinate Court)."
                else:
                    return "District Civil Court (Principal Civil Court)."
            return "District Civil Court based on nature of dispute."
        elif case_type == "Consumer":
            if amount:
                if amount <= 5000000:
                    return "District Consumer Disputes Redressal Commission (DCDRC)."
                elif amount <= 20000000:
                    return "State Consumer Disputes Redressal Commission (SCDRC)."
                else:
                    return "National Consumer Disputes Redressal Commission (NCDRC)."
            return "Consumer Forum – Specify claim amount."
        elif case_type == "Family":
            return "Family Court in your district – Handles divorce, custody, and maintenance."
        elif case_type == "Labour":
            return "Labour Court or Industrial Tribunal depending on the dispute."
        elif case_type == "Property":
            return "Civil Court or Revenue Court depending on type of property dispute."
        return "Consult a legal expert or your District Civil Court."

    if submitted:
        suggestion = recommend_court(case_type, amount_involved, special_case)
        st.markdown("### 🧭 Recommended Court")
        st.success(suggestion)

# ========== STYLE ==========
st.markdown("""
<style>
div.stAlert > div {
    font-size: 16px;
}
section.main > div {
    padding-top: 15px;
}
</style>
""", unsafe_allow_html=True)


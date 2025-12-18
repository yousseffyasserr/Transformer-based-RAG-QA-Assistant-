import streamlit as st
import os, re, unicodedata, torch, torch.nn as nn, numpy as np
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Transformer-based RAG QA-Assistant", page_icon="📄", layout="wide")
st.title("📄Transformer-based RAG QA-Assistant")

# ---------------------- FILE FUNCTIONS ----------------------
def getFileType(file_name):
    ext = file_name.lower().split(".")[-1]
    if ext == "pdf": return "pdf"
    if ext in ["doc", "docx"]: return "docx"
    raise ValueError("Unsupported file type")

def pdfRead(file_path):
    reader = PdfReader(file_path)
    return " ".join([page.extract_text() for page in reader.pages])

def docsRead(file_path):
    doc = Document(file_path)
    return " ".join([para.text for para in doc.paragraphs])

def cleanText(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return unicodedata.normalize("NFKC", text.strip().lower())

def ingestFile(file_path):
    file_type = getFileType(file_path)
    raw = pdfRead(file_path) if file_type=="pdf" else docsRead(file_path)
    text = cleanText(raw)
    metadata = {
        "filename": os.path.basename(file_path),
        "length": len(text),
        "num_words": len(text.split())
    }
    return text, metadata

def chunk_text(text, chunk_size=800, overlap=120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += (chunk_size - overlap)
    return chunks

# ---------------------- EMBEDDINGS ----------------------
@st.cache_resource
def loadEmbeddingModel():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(model, chunks):
    vectors = model.encode(chunks, convert_to_numpy=True)
    return vectors.astype("float32")

def embed_query(model, query):
    vec = model.encode([query], convert_to_numpy=True)
    return vec.astype("float32")

def retrieve(query, embeddings, chunks, embed_model, k=5):
    q_vec = embed_query(embed_model, query)
    sims = cosine_similarity(q_vec, embeddings)[0]
    idxs = sims.argsort()[::-1][:k]
    results = []
    for i in idxs:
        results.append({"chunk": chunks[i], "score": float(sims[i]), "id": i})
    return results

def build_context(results):
    context = ""
    for r in results:
        context += f"[Score: {round(r['score'],3)}] {r['chunk']}\n\n"
    return context

# ---------------------- TRANSFORMER ----------------------
@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_answer_transformer(model, tokenizer, query, context, max_tokens=150):
    prompt = f"Use ONLY this context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.replace(prompt, "").strip()

# ---------------------- SEQ2SEQ GRU ----------------------
class AttentionGRU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, hidden_top, encoder_outputs):
        proj = self.W(encoder_outputs)
        score = torch.bmm(proj, hidden_top.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Seq2SeqGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, n_layers=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        self.attn = AttentionGRU(hid_dim)
        self.fc_out = nn.Linear(hid_dim*2, vocab_size)
        self.hid_dim, self.n_layers, self.pad_idx = hid_dim, n_layers, pad_idx
    def encode(self, src_ids):
        embedded = self.embedding(src_ids)
        enc_outputs, h = self.encoder(embedded)
        return enc_outputs, h
    def decode_step(self, input_ids, hidden, encoder_outputs):
        embedded = self.embedding(input_ids)
        output, hidden = self.decoder(embedded, hidden)
        dec_top = output.squeeze(1)
        h_top = hidden[-1]
        context, _ = self.attn(h_top, encoder_outputs)
        combined = torch.cat([dec_top, context], dim=1)
        logits = self.fc_out(combined)
        return logits, hidden
    def forward(self, src_ids, trg_ids, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encode(src_ids)
        batch_size, max_len = src_ids.size(0), trg_ids.size(1)
        outputs = torch.zeros(batch_size, max_len, self.fc_out.out_features, device=src_ids.device)
        input_tok = trg_ids[:,0].unsqueeze(1)
        for t in range(1, max_len):
            logits, hidden = self.decode_step(input_tok, hidden, encoder_outputs)
            outputs[:, t, :] = logits
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1).unsqueeze(1)
            input_tok = trg_ids[:, t].unsqueeze(1) if teacher_force else top1
        return outputs

@st.cache_resource
def load_gru(_tokenizer):
    vocab_size = getattr(_tokenizer, "vocab_size", 30522)
    pad_idx = _tokenizer.pad_token_id or 0
    return Seq2SeqGRU(vocab_size=vocab_size, pad_idx=pad_idx)

# ---------------------- ENHANCED GUI ----------------------
st.header("📁 Upload Document")
uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf","docx"])

if uploaded:
    col1, col2 = st.columns([1,2])

    # --- Left Column: File Info ---
    with col1:
        file_path = f"temp_{uploaded.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())
        text, meta = ingestFile(file_path)
        st.success(f"Loaded **{meta['filename']}** ({meta['num_words']} words)")
        st.markdown(f"**Total characters:** {meta['length']}")
        st.markdown(f"**Number of words:** {meta['num_words']}")

    # --- Right Column: QA ---
    with col2:
        chunks = chunk_text(text)
        st.info(f"Created {len(chunks)} chunks")
        with st.spinner("Generating embeddings..."):
            embed_model = loadEmbeddingModel()
            embeddings = embed_chunks(embed_model, chunks)
        st.success("Embeddings ready ✅")

        st.subheader("❓ Ask a Question")
        query = st.text_input("Type your question here")
        model_choice = st.radio("Choose model", ["Transformer","Seq2Seq GRU"],
                                help="Transformer = pretrained TinyLlama\nSeq2Seq GRU = baseline (untrained)")

        if query:
            with st.spinner("Retrieving relevant chunks..."):
                results = retrieve(query, embeddings, chunks, embed_model)
                context = build_context(results)
            with st.spinner("Generating answer..."):
                if model_choice=="Transformer":
                    model, tokenizer = load_llm()
                    answer = generate_answer_transformer(model, tokenizer, query, context)
                else:
                    model, tokenizer = load_llm()
                    gru_model = load_gru(tokenizer).to("cuda" if torch.cuda.is_available() else "cpu")
                    gru_model.eval()
                    sep_token = tokenizer.sep_token or tokenizer.eos_token
                    input_text = context + f" {sep_token} " + query
                    tokens = tokenizer.encode(input_text, truncation=True, max_length=256)
                    src = torch.tensor([tokens], dtype=torch.long).to(gru_model.embedding.weight.device)
                    trg = src.clone()
                    with torch.no_grad():
                        outputs = gru_model(src, trg, teacher_forcing_ratio=0.0)
                        answer_ids = [outputs[0,i].argmax().item() for i in range(outputs.size(1)) if outputs[0,i].argmax().item() != tokenizer.eos_token_id]
                        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

            st.subheader("📌 Answer")
            st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px'>{answer}</div>", unsafe_allow_html=True)

            with st.expander("🔍 Retrieved Chunks"):
                for r in results:
                    st.markdown(f"**Score:** {round(r['score'],4)}")
                    st.markdown(f"{r['chunk']}")
                    st.markdown("---")

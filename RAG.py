# seq2seq_lstm_rag_fixed.py
# Updated single-file RAG prototype + Seq2Seq LSTM baseline (padding fixes + robustness)
# Key fixes:
# - Ensure tokenizer pad/bos/eos exist BEFORE constructing LSTM embeddings
# - Use len(tokenizer) for vocab_size so padding_idx < vocab_size
# - Lazy-load huge transformer only if requested
# - Use torch.optim.AdamW with lower LR
# - Sanitize repeated punctuation
# - Sampling (temp + top-k) decoding to avoid collapse
# - Clear GPU cache before training LSTM if transformer was loaded

!pip install pypdf python-docx numpy faiss-cpu sentence-transformers
!pip install transformers accelerate torch
!pip install bitsandbytes


import os
import re
import unicodedata
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ---------------------------
# Basic file reading + cleaning
# ---------------------------

def getFileType(file_name: str) -> str:
    ext = file_name.lower().split('.')[-1]
    if ext == 'pdf':
        return 'pdf'
    if ext in ['doc', 'docx']:
        return 'docx'
    raise ValueError('Unsupported file type')

def pdfRead(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text.append(page_text)
    if not text:
        raise ValueError('No text could be extracted from the PDF.')
    return ' '.join(text)

def docsRead(file_path: str) -> str:
    doc = Document(file_path)
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    if not text:
        raise ValueError('No text could be extracted from the DOCX.')
    return ' '.join(text)

def cleanText(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^\x00-\x7F]+", '', text)
    return unicodedata.normalize('NFKC', text.strip())

def ingestFile(file_path: str) -> Tuple[str, Dict]:
    file_type = getFileType(file_path)
    raw = pdfRead(file_path) if file_type == 'pdf' else docsRead(file_path)
    text = cleanText(raw)
    metadata = {
        'filename': os.path.basename(file_path),
        'length': len(text),
        'num_words': len(text.split())
    }
    return text, metadata

# ---------------------------
# Chunking
# ---------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text.strip():
        return []
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

# ---------------------------
# Embeddings
# ---------------------------
@torch.no_grad()
def loadEmbeddingModel():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(model, chunks: List[str]):
    if not chunks:
        return np.zeros((1, model.get_sentence_embedding_dimension()), dtype='float32')
    vectors = model.encode(chunks, convert_to_numpy=True)
    return vectors.astype('float32')

def embed_query(model, query: str):
    vec = model.encode([query], convert_to_numpy=True)
    return vec.astype('float32')

# ---------------------------
# Retrieval
# ---------------------------

def retrieve(query: str, embeddings: np.ndarray, chunks: List[str], embed_model, k: int = 5):
    if len(chunks) == 0 or embeddings is None:
        return []
    q_vec = embed_query(embed_model, query)
    sims = cosine_similarity(q_vec, embeddings)[0]
    idxs = sims.argsort()[::-1][:k]
    results = []
    for i in idxs:
        results.append({'chunk': chunks[i], 'score': float(sims[i]), 'id': int(i)})
    return results

def build_context(results, max_chars: int = 1500) -> str:
    context = ''
    total_chars = 0
    for r in results:
        chunk = r['chunk']
        if total_chars + len(chunk) > max_chars:
            chunk = chunk[:max_chars - total_chars]
        context += f"[Score: {round(r['score'],3)}]\n{chunk}\n\n"
        total_chars += len(chunk)
        if total_chars >= max_chars:
            break
    return context

# ---------------------------
# Transformer LLM loader
# ---------------------------
mistral_model_name = 'mistralai/Mistral-7B-v0.1'

def ensure_special_tokens(tokenizer: AutoTokenizer) -> bool:
    """
    Ensure pad/bos/eos exist. If pad token is added, tokenizer vocab increases.
    Returns True if tokenizer changed (so caller may need to re-evaluate vocab-dependent objects).
    """
    changed = False
    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        changed = True
    if getattr(tokenizer, 'bos_token', None) is None and getattr(tokenizer, 'cls_token', None) is not None:
        tokenizer.bos_token = tokenizer.cls_token
    if getattr(tokenizer, 'eos_token', None) is None and getattr(tokenizer, 'sep_token', None) is not None:
        tokenizer.eos_token = tokenizer.sep_token
    return changed

def load_transformer_llm():
    tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
    changed = ensure_special_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        mistral_model_name,
        device_map='auto',
        torch_dtype=torch.float16
    )
    # If tokenizer changed and model embeddings need resizing, we'd handle it here. For Mistral it's typically fine.
    return model, tokenizer

def generate_answer_transformer(model, tokenizer, query: str, context: str, max_tokens: int = 150):
    if not context.strip():
        return 'No relevant context found in the document.'
    prompt = f"Use ONLY this context:\n{context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = text.split('Answer:')[-1].strip()
    answer = re.split(r'Question:', answer)[0].strip()
    return answer

# ---------------------------
# Seq2Seq LSTM with Attention
# ---------------------------
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden, encoder_outputs):
        score = torch.bmm(self.W(encoder_outputs), hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, n_layers=2, pad_idx=0):
        super().__init__()
        # vocab_size and pad_idx must be consistent: 0 <= pad_idx < vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.attn = LuongAttention(hid_dim)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def encode(self, src_ids):
        embedded = self.embedding(src_ids)
        enc_outputs, (h, c) = self.encoder(embedded)
        return enc_outputs, (h, c)

    def decode_step(self, input_ids, hidden, encoder_outputs):
        embedded = self.embedding(input_ids)
        output, hidden = self.decoder(embedded, hidden)
        dec_top = output.squeeze(1)
        h_top = hidden[0][-1]
        context, attn_weights = self.attn(h_top, encoder_outputs)
        combined = dec_top + context
        logits = self.fc_out(combined)
        return logits, hidden, attn_weights

    def forward(self, src_ids, trg_ids=None, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encode(src_ids)
        batch_size = src_ids.size(0)
        max_len = trg_ids.size(1) if trg_ids is not None else 128
        outputs = torch.zeros(batch_size, max_len, self.fc_out.out_features, device=src_ids.device)

        input_tok = trg_ids[:, 0].unsqueeze(1) if trg_ids is not None else torch.zeros((batch_size,1), dtype=torch.long, device=src_ids.device)
        for t in range(1, max_len):
            logits, hidden, _ = self.decode_step(input_tok, hidden, encoder_outputs)
            outputs[:, t, :] = logits
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1).unsqueeze(1)
            input_tok = trg_ids[:, t].unsqueeze(1) if (trg_ids is not None and teacher_force) else top1
        return outputs

# ---------------------------
# Dataset & training utilities (self-supervised pairing from chunks)
# ---------------------------
class ChunkPairDataset(Dataset):
    def __init__(self, tokenizer, chunks: List[str], max_src_len=256, max_trg_len=128):
        self.tokenizer = tokenizer
        self.pairs = []
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        bos = getattr(tokenizer, 'bos_token_id', None)
        for ch in chunks:
            ch_sanitized = re.sub(r'[:]{2,}', ':', ch)  # collapse repeated colons
            tokens = tokenizer.encode(ch_sanitized, add_special_tokens=False)
            if len(tokens) < 10:
                continue
            split = len(tokens) // 2
            src = tokens[:split][: (max_src_len - (1 if bos is not None else 0))]
            trg = tokens[split:split + (max_trg_len - (1 if bos is not None else 0))]
            if bos is not None:
                src_ids = [bos] + src
                trg_ids = [bos] + trg
            else:
                src_ids = src
                trg_ids = trg
            self.pairs.append((src_ids, trg_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch, pad_idx=0, max_src_len=256, max_trg_len=128):
    srcs, trgs = zip(*batch)
    def pad(seq, maxlen):
        seq = seq[:maxlen]
        return seq + [pad_idx] * (maxlen - len(seq))
    src_batch = [pad(s, max_src_len) for s in srcs]
    trg_batch = [pad(t, max_trg_len) for t in trgs]
    src_tensor = torch.tensor(src_batch, dtype=torch.long)
    trg_tensor = torch.tensor(trg_batch, dtype=torch.long)
    return src_tensor, trg_tensor

def train_seq2seq(model: Seq2SeqLSTM, tokenizer, chunks: List[str], device='cpu',
                   epochs=3, batch_size=8, lr=1e-4):
    """
    Train seq2seq LSTM. Uses AdamW with a smaller LR and gradient clipping.
    Important: ensure tokenizer has pad token before calling this function.
    """
    model.to(device)
    dataset = ChunkPairDataset(tokenizer, chunks)
    if len(dataset) == 0:
        raise ValueError('No training pairs generated from chunks. Provide a longer document or more chunks.')
    pad_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, pad_idx=pad_idx,
                                                        max_src_len=256, max_trg_len=128))
    optim = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, trg in loader:
            src = src.to(device)
            trg = trg.to(device)
            optim.zero_grad()
            outputs = model(src, trg, teacher_forcing_ratio=0.6)
            outputs = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
            targets = trg[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch} - Loss: {avg:.4f}")
    return model

# ---------------------------
# Inference: sampling decoding to avoid collapse
# ---------------------------
def generate_answer_lstm(model: Seq2SeqLSTM, tokenizer, query: str, context: str, max_tokens: int = 150, device='cpu') -> str:
    if not context.strip():
        return 'No relevant context found in the document.'
    prompt = f"{context}\nQ: {query}\nA:"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)[:250]
    src = torch.tensor([input_ids], dtype=torch.long, device=device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encode(src)
        start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else (tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
        cur = torch.tensor([[start_token]], dtype=torch.long, device=device)
        generated = []
        for _ in range(max_tokens):
            logits, hidden, _ = model.decode_step(cur, hidden, encoder_outputs)
            logits = logits.squeeze(0)
            probs = torch.softmax(logits / 0.8, dim=0)  # temperature
            topk = min(50, probs.size(0))
            vals, inds = torch.topk(probs, topk)
            vals = vals.cpu().numpy()
            inds = inds.cpu().numpy()
            if vals.sum() <= 0:
                next_id = int(inds[0])
            else:
                next_id = int(np.random.choice(inds, p=(vals / vals.sum())))
            if next_id == (tokenizer.eos_token_id or -1):
                break
            generated.append(next_id)
            cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        try:
            text = tokenizer.decode(generated, skip_special_tokens=True)
        except Exception:
            text = ' '.join([str(t) for t in generated])
        answer = text.strip()
        return answer

# ---------------------------
# Simple evaluation utilities
# ---------------------------
def context_overlap_ratio(answer: str, retrieved_chunks: List[Dict]) -> float:
    ans_tokens = answer.split()
    ctx = ' '.join([r['chunk'] for r in retrieved_chunks])
    ctx_tokens = set(ctx.split())
    if len(ans_tokens) == 0:
        return 0.0
    overlap = sum(1 for t in ans_tokens if t in ctx_tokens)
    return overlap / len(ans_tokens)

def sbert_cosine_similarity(embed_model, answer: str, retrieved_chunks: List[Dict]) -> float:
    if not answer.strip() or len(retrieved_chunks) == 0:
        return 0.0
    ans_emb = embed_query(embed_model, answer)
    ctx = ' '.join([r['chunk'] for r in retrieved_chunks])
    ctx_emb = embed_query(embed_model, ctx)
    sim = cosine_similarity(ans_emb, ctx_emb)[0][0]
    return float(sim)

# ---------------------------
# Integration: Gradio UI + switching logic
# ---------------------------

CACHE = {
    'embed_model': None,
    'transformer_model': None,
    'transformer_tokenizer': None,
    'lstm_model': None,
    'lstm_tokenizer': None,
}

def ensure_embed_model():
    if CACHE['embed_model'] is None:
        CACHE['embed_model'] = loadEmbeddingModel()
    return CACHE['embed_model']

def ensure_transformer():
    if CACHE['transformer_model'] is None or CACHE['transformer_tokenizer'] is None:
        model, tokenizer = load_transformer_llm()
        CACHE['transformer_model'] = model
        CACHE['transformer_tokenizer'] = tokenizer
    return CACHE['transformer_model'], CACHE['transformer_tokenizer']

def ensure_lstm(tokenizer, device='cpu'):
    """
    Build LSTM model AFTER ensuring tokenizer special tokens exist and vocab size is known.
    """
    if CACHE['lstm_model'] is None:
        # Ensure tokenizer has pad token and special tokens
        changed = ensure_special_tokens(tokenizer)
        # Recompute vocab size *after* any token additions
        vocab_size = len(tokenizer)
        pad_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # sanity check: pad_idx must be within [0, vocab_size)
        if not (0 <= pad_idx < vocab_size):
            # fix by mapping pad token string to id (re-add if necessary)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_idx = tokenizer.pad_token_id
            vocab_size = len(tokenizer)
            if not (0 <= pad_idx < vocab_size):
                # final fallback
                pad_idx = 0
        lstm = Seq2SeqLSTM(vocab_size=vocab_size, emb_dim=256, hid_dim=512, n_layers=2, pad_idx=pad_idx)
        CACHE['lstm_model'] = lstm
        CACHE['lstm_tokenizer'] = tokenizer
    return CACHE['lstm_model']

# Main QA function used by Gradio
def qa_interface(file, query, model_choice='Transformer', train_lstm=False, lstm_epochs=2):
    try:
        file_path = file.name
        # Ingest
        text, meta = ingestFile(file_path)
        chunks = chunk_text(text)
        if not chunks:
            return 'No text could be extracted from this document.', 'No chunks available.'

        # Embeddings
        embed_model = ensure_embed_model()
        embeddings = embed_chunks(embed_model, chunks)

        # Retrieve
        results = retrieve(query, embeddings, chunks, embed_model)
        context = build_context(results)

        transformer_model = transformer_tokenizer = None
        if model_choice == 'Transformer':
            transformer_model, transformer_tokenizer = ensure_transformer()
            start = time.time()
            answer = generate_answer_transformer(transformer_model, transformer_tokenizer, query, context)
            latency = time.time() - start
            mode = 'Transformer'
        else:
            # Seq2Seq LSTM
            # Get tokenizer (try cache first; if absent, load transformer tokenizer - note: large)
            if CACHE.get('transformer_tokenizer') is None:
                # load tokenizer (and model) lazily; acceptable for prototype
                _, tokenizer = ensure_transformer()
            else:
                tokenizer = CACHE['transformer_tokenizer']

            # ensure special tokens exist before building LSTM
            ensure_special_tokens(tokenizer)
            lstm_model = ensure_lstm(tokenizer)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # If transformer model exists on GPU, free it to avoid OOM
            if train_lstm and CACHE.get('transformer_model') is not None:
                try:
                    del CACHE['transformer_model']
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if train_lstm:
                print('Training LSTM on document chunks (this may take a while)...')
                lstm_model = train_seq2seq(lstm_model, tokenizer, chunks, device=device, epochs=lstm_epochs)
                CACHE['lstm_model'] = lstm_model

            start = time.time()
            answer = generate_answer_lstm(lstm_model, tokenizer, query, context, max_tokens=150, device=device)
            latency = time.time() - start
            mode = 'Seq2Seq LSTM'

        overlap = context_overlap_ratio(answer, results)
        sbert_sim = sbert_cosine_similarity(embed_model, answer, results)

        retrieved_text = ''
        for r in results:
            retrieved_text += f"Score: {round(r['score'],4)}\n{r['chunk']}\n---\n"

        meta_out = (
            f"Model: {mode}\nLatency: {latency:.3f}s\nContext-Overlap: {overlap:.3f}\nSBERT-sim: {sbert_sim:.3f}\n"
        )

        return answer, meta_out + '\nRetrieved Chunks:\n' + retrieved_text
    except Exception as e:
        return f'Error: {str(e)}', ''

# ---------------------------
# Gradio UI
# ---------------------------
iface = gr.Interface(
    fn=qa_interface,
    inputs=[
        gr.File(file_types=['.pdf', '.docx'], label='Upload Document'),
        gr.Textbox(lines=2, placeholder='Enter your question here', label='Your Question'),
        gr.Radio(['Transformer', 'Seq2Seq LSTM'], value='Transformer', label='Choose generator'),
        gr.Checkbox(label='Train LSTM on this document before inference (slow)', value=False),
        gr.Slider(minimum=1, maximum=10, value=2, step=1, label='LSTM epochs (if training)')
    ],
    outputs=[
        gr.Textbox(label='Answer', lines=6, interactive=False),
        gr.Textbox(label='Diagnostics + Retrieved Chunks', lines=20, interactive=False)
    ],
    title='🤖 Transformer-based RAG QA-Assistant with Seq2Seq LSTM Baseline',
    description=(
        'Upload a PDF or DOCX document and ask questions. '
        'Switch between the Transformer (Mistral) and a Seq2Seq LSTM baseline. '
        'If you choose Seq2Seq LSTM, you may optionally train it on the document chunks before inference. '
    ),
)

if __name__ == '__main__':
    iface.launch(share=True)

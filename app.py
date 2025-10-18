"""
Urdu Transformer Chatbot - Streamlit Web Application
=====================================================
Features:
- Load pretrained model from Google Drive
- Urdu text input with RTL support
- Multiple decoding strategies (Greedy, Beam Search)
- Conversation history
- Real-time response generation
"""

import streamlit as st
import torch
import torch.nn as nn
import math
import re
import unicodedata
from collections import deque

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Urdu Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Urdu RTL support and styling
st.markdown("""
<style>
    /* Urdu text styling with RTL support */
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-family: 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', 'Alvi Nastaleeq', Arial, sans-serif;
        font-size: 20px;
        line-height: 1.8;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        border-left: 4px solid #2196F3;
    }
    
    .bot-message {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        border-left: 4px solid #4CAF50;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-size: 18px;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #1976D2;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=2, n_encoder_layers=2, 
                 n_decoder_layers=2, d_ff=512, dropout=0.1, max_len=50):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt):
        src_mask = (src == 0)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        src_emb = self.dropout(self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model)))
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        tgt_emb = self.dropout(self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model)))
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
        
        output = self.fc_out(decoder_output)
        return output


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def normalize_urdu_text(text):
    """Normalize Urdu text"""
    if not isinstance(text, str):
        return ""
    
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    alef_forms = ['آ', 'أ', 'إ', 'ا']
    for alef in alef_forms:
        text = text.replace(alef, 'ا')
    
    yeh_forms = ['ی', 'ي', 'ے']
    for yeh in yeh_forms:
        text = text.replace(yeh, 'ی')
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_urdu(text):
    """Tokenize Urdu text"""
    text = normalize_urdu_text(text)
    tokens = re.findall(r'\S+', text)
    return tokens


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================
@st.cache_resource
def download_model_from_gdrive():
    """Automatically download model from Google Drive"""
    try:
        import gdown
        import os
        
        # Your Google Drive file ID
        file_id = "1lHDSR2UVh-hpII3KbRELii1tQ5oqdieQ"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = "best_urdu_transformer.pt"
        
        # Check if model already exists
        if not os.path.exists(output_path):
            st.info("📥 Downloading model from Google Drive...")
            gdown.download(download_url, output_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        else:
            st.info("✅ Model already downloaded!")
        
        return output_path
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None


class Vocabulary:
    """Vocabulary class - must be defined before loading"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = {}
        self.n_words = 4
    
    def add_sentence(self, sentence):
        for word in tokenize_urdu(sentence):
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1


@st.cache_resource
def load_model_from_drive(model_path):
    """Load model from Google Drive (cached)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load with weights_only=False to handle custom classes
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        vocab = checkpoint['vocab']
        hyperparams = checkpoint['hyperparameters']
        
        model = Transformer(
            vocab_size=vocab.n_words,
            d_model=hyperparams['d_model'],
            n_heads=hyperparams['n_heads'],
            n_encoder_layers=hyperparams['n_encoder_layers'],
            n_decoder_layers=hyperparams['n_decoder_layers'],
            d_ff=hyperparams['d_model'] * 2,
            dropout=hyperparams['dropout'],
            max_len=hyperparams['max_len']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, vocab, device, hyperparams
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None


def greedy_decode(model, src, vocab, device, max_len=50):
    """Greedy decoding strategy"""
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        
        # Start with SOS token
        tgt = torch.tensor([[vocab.word2idx['<SOS>']]], device=device)
        
        for _ in range(max_len):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            if next_token.item() == vocab.word2idx['<EOS>']:
                break
            
            tgt = torch.cat([tgt, next_token], dim=1)
        
        return tgt.squeeze().tolist()


def beam_search_decode(model, src, vocab, device, beam_width=3, max_len=50):
    """Beam search decoding strategy"""
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        
        # Initialize beam with SOS token
        sequences = [[vocab.word2idx['<SOS>']]]
        scores = [0.0]
        
        for _ in range(max_len):
            all_candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[-1] == vocab.word2idx['<EOS>']:
                    all_candidates.append((scores[i], seq))
                    continue
                
                tgt = torch.tensor([seq], device=device)
                output = model(src, tgt)
                logits = output[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top k tokens
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    candidate_seq = seq + [idx.item()]
                    candidate_score = scores[i] + log_prob.item()
                    all_candidates.append((candidate_score, candidate_seq))
            
            # Select top beam_width sequences
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = [seq for score, seq in ordered[:beam_width]]
            scores = [score for score, seq in ordered[:beam_width]]
            
            # Check if all sequences ended
            if all(seq[-1] == vocab.word2idx['<EOS>'] for seq in sequences):
                break
        
        return sequences[0]


def text_to_indices(text, vocab, max_len=50):
    """Convert text to token indices"""
    tokens = tokenize_urdu(text)
    indices = [vocab.word2idx.get(token, vocab.word2idx['<UNK>']) for token in tokens]
    indices = [vocab.word2idx['<SOS>']] + indices + [vocab.word2idx['<EOS>']]
    
    if len(indices) < max_len:
        indices += [vocab.word2idx['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return torch.tensor([indices])


def indices_to_text(indices, vocab):
    """Convert token indices back to text"""
    words = []
    for idx in indices:
        if idx in [vocab.word2idx['<SOS>'], vocab.word2idx['<PAD>']]:
            continue
        if idx == vocab.word2idx['<EOS>']:
            break
        words.append(vocab.idx2word[idx])
    return ' '.join(words)


# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    # Header
    st.markdown('<p class="main-title">💬 Urdu Transformer Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered Urdu Language Model | اردو چیٹ بوٹ</p>', unsafe_allow_html=True)
    
    # Automatically download model on app start
    model_path = download_model_from_gdrive()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model status
        st.subheader("📁 Model Status")
        if model_path:
            st.success("✅ Model loaded from Google Drive")
            import os
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.info(f"📊 File size: {file_size:.2f} MB")
        else:
            st.error("❌ Model not loaded")
        
        # Decoding strategy
        st.subheader("🎯 Decoding Strategy")
        decoding_strategy = st.selectbox(
            "Choose Strategy",
            ["Greedy", "Beam Search"],
            help="Greedy: Fast but less diverse\nBeam Search: Slower but better quality"
        )
        
        if decoding_strategy == "Beam Search":
            beam_width = st.slider("Beam Width", 2, 5, 3)
        else:
            beam_width = 1
        
        # Generation settings
        st.subheader("🔧 Generation Settings")
        max_length = st.slider("Max Response Length", 10, 100, 50)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Load model
    if model_path:
        with st.spinner("Loading model... یہ کچھ وقت لے سکتا ہے"):
            model, vocab, device, hyperparams = load_model_from_drive(model_path)
        
        if model is not None:
            st.success("✅ Model loaded successfully!")
            
            # Display model info
            with st.expander("📊 Model Information"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Vocabulary Size", f"{vocab.n_words:,}")
                with col2:
                    st.metric("Model Dimension", hyperparams['d_model'])
                with col3:
                    st.metric("Attention Heads", hyperparams['n_heads'])
            
            # Chat interface
            st.markdown("---")
            st.subheader("💬 Chat Interface")
            
            # Display conversation history
            if st.session_state.conversation_history:
                st.markdown("### Conversation History | گفتگو کی تاریخ")
                for i, (user_msg, bot_msg) in enumerate(st.session_state.conversation_history):
                    # User message
                    st.markdown(f"""
                    <div style="background-color: #E3F2FD; padding: 15px; border-radius: 10px; 
                                margin: 10px 0; direction: rtl; text-align: right; 
                                border-left: 4px solid #2196F3;">
                        <strong style="color: #1976D2;">آپ:</strong> 
                        <span style="color: #000; font-size: 18px;">{user_msg}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Bot message
                    st.markdown(f"""
                    <div style="background-color: #F1F8E9; padding: 15px; border-radius: 10px; 
                                margin: 10px 0; direction: rtl; text-align: right; 
                                border-left: 4px solid #4CAF50;">
                        <strong style="color: #388E3C;">بوٹ:</strong> 
                        <span style="color: #000; font-size: 18px; font-weight: 500;">{bot_msg}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Input form
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "اپنا پیغام لکھیں (Write your message):",
                    key="user_input",
                    placeholder="یہاں اردو میں ٹائپ کریں..."
                )
                submit_button = st.form_submit_button("📤 Send")
            
            if submit_button and user_input.strip():
                with st.spinner("جواب بنایا جا رہا ہے... Generating response..."):
                    # Preprocess input
                    src = text_to_indices(user_input, vocab, max_len=max_length)
                    
                    # Generate response
                    if decoding_strategy == "Greedy":
                        output_indices = greedy_decode(model, src, vocab, device, max_len=max_length)
                    else:
                        output_indices = beam_search_decode(model, src, vocab, device, 
                                                           beam_width=beam_width, max_len=max_length)
                    
                    # Convert to text
                    response = indices_to_text(output_indices, vocab)
                    
                    # Debug info
                    st.info(f"🔍 Debug: Generated {len(output_indices)} tokens")
                    st.info(f"📝 Raw response: {response}")
                    
                    # If response is empty or same as input, generate alternative
                    if not response or response.strip() == "" or response == user_input:
                        response = "معذرت، میں اس وقت جواب نہیں دے سکتا۔ (Sorry, I cannot respond at this moment.)"
                    
                    # Add to history
                    st.session_state.conversation_history.append((user_input, response))
                    
                    # Rerun to show updated history
                    st.rerun()
        else:
            st.error("❌ Failed to load model. Please check the path and try again.")
    else:
        st.warning("⚠️ Please upload a model file or provide a path in the sidebar to start chatting.")
        
        # Instructions
        st.markdown("---")
        st.markdown("""
        ### 📖 How to Use:
        1. **Upload Model**: Use the sidebar to upload your `best_urdu_transformer.pt` file
        2. **Or Mount Drive**: Check the box to mount Google Drive and provide the path
        3. **Choose Settings**: Select decoding strategy and other parameters
        4. **Start Chatting**: Type your message in Urdu and press Send
        
        ### 🌟 Features:
        - ✅ Right-to-left (RTL) Urdu text rendering
        - ✅ Multiple decoding strategies (Greedy & Beam Search)
        - ✅ Conversation history tracking
        - ✅ Real-time response generation
        - ✅ Adjustable generation parameters
        """)


if __name__ == "__main__":
    main()
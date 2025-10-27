import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary

# ------------------------------------------------
# 🔧 PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="🖼️ Image Captioning", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1a1a2e, #16213e);
        color: white;
    }
    .stButton>button {
        background-color: #8338ec;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ AI Image Captioning with PyTorch")

# ------------------------------------------------
# 📁 PATHS
# ------------------------------------------------
VOCAB_PATH = "data/vocab.pkl"
ENCODER_PATH = "models/encoder-5-3000.pkl"
DECODER_PATH = "models/decoder-5-3000.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# 🧠 LOADERS
# ------------------------------------------------
@st.cache_resource
def load_vocab():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    return vocab

@st.cache_resource
def load_model():
    vocab = load_vocab()
    encoder = EncoderCNN(embed_size=256).eval().to(device)
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1).eval().to(device)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    return encoder, decoder, vocab

encoder, decoder, vocab = load_model()

# ------------------------------------------------
# 🎨 IMAGE TRANSFORM
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ------------------------------------------------
# 🧾 CAPTION GENERATION
# ------------------------------------------------
def generate_caption(image, encoder, decoder, vocab):
    image_tensor = transform(image).unsqueeze(0).to(device)
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        caption.append(word)
    return ' '.join(caption), feature

# ------------------------------------------------
# 📊 VISUALIZATIONS
# ------------------------------------------------
def visualize_features(feature):
    """Visualize the encoder output as a heatmap (if spatial)."""
    if len(feature.shape) == 4:  # [1, C, H, W]
        feature_np = feature.squeeze().detach().cpu().numpy()
        fig, ax = plt.subplots()
        sns.heatmap(np.mean(feature_np, axis=0), ax=ax, cmap="magma")
        st.pyplot(fig)

def show_wordcloud(history):
    """Show word cloud of all generated captions."""
    if history:
        text = " ".join([x['caption'] for x in history])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

# ------------------------------------------------
# ⚙️ MAIN UI
# ------------------------------------------------
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### Model Info")
    st.info(f"Running on: **{'GPU' if torch.cuda.is_available() else 'CPU'}**")
    st.caption("Embedding size: 256 | Hidden size: 512 | Layers: 1")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------
# 🚀 GENERATE CAPTION
# ------------------------------------------------
if uploaded_file is not None:
    if st.button("✨ Generate Caption"):
        with st.spinner("Generating caption..."):
            for p in range(100):
                time.sleep(0.005)
                st.progress(p + 1)
            
            caption, feature = generate_caption(image, encoder, decoder, vocab)
            st.session_state.history.append({"caption": caption, "filename": uploaded_file.name})

        st.success("✅ Caption Generated!")
        st.markdown(f"### 🗨️ {caption.capitalize()}")

        # Step-by-step visualization
        with st.expander("🧩 Step-by-step generation"):
            for i, word in enumerate(caption.split()):
                st.write(f"**Step {i+1}:** {word}")
                time.sleep(0.1)

        # Feature visualization
        with st.expander("🔥 Encoder Feature Map"):
            visualize_features(feature)

# ------------------------------------------------
# 🗂️ HISTORY + WORD CLOUD
# ------------------------------------------------
if st.session_state.history:
    st.markdown("## 🧠 Caption History")
    for item in st.session_state.history[-5:]:
        st.write(f"🖼️ **{item['filename']}** → {item['caption']}")

    st.download_button(
        "💾 Download All Captions",
        "\n".join([f"{x['filename']}: {x['caption']}" for x in st.session_state.history]),
        file_name="captions.txt"
    )

    st.markdown("## ☁️ Word Cloud of Generated Captions")
    show_wordcloud(st.session_state.history)

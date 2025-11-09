import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pickle
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary
import os
import math
import re

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("🖼️ Image Captioning with PyTorch")

VOCAB_PATH = "data/vocab.pkl"
ENCODER_PATH = "models/encoder-5-3000.pkl"
DECODER_PATH = "models/decoder-5-3000.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_vocab():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    return vocab

# Load models
@st.cache_resource
def load_model():
    vocab = load_vocab()
    encoder = EncoderCNN(embed_size=256).eval().to(device)
    decoder = DecoderRNN(
        embed_size=256,
        hidden_size=512,
        vocab_size=len(vocab),
        num_layers=1
    ).eval().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    
    return encoder, decoder, vocab

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

try:
    encoder, decoder, vocab = load_model()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

def post_process_caption(caption):
    """Post-process caption for better readability"""
    if not caption or caption.strip() == "":
        return "No caption generated"
    
    # Capitalize first letter
    caption = caption.strip().capitalize()
    
    # Fix common issues
    caption = caption.replace(' a a ', ' a ').replace(' an a ', ' an ')
    caption = caption.replace(' .', '.').replace(' ,', ',')
    caption = caption.replace(' ,', ',').replace('  ', ' ')
    
    # Remove duplicate words
    words = caption.split()
    if len(words) > 1:
        cleaned_words = [words[0]]
        for i in range(1, len(words)):
            if words[i] != words[i-1]:
                cleaned_words.append(words[i])
        caption = ' '.join(cleaned_words)
    
    # Fix common grammatical issues
    caption = re.sub(r'\ba a\b', 'a', caption)
    caption = re.sub(r'\ban an\b', 'an', caption)
    caption = re.sub(r'\bthe the\b', 'the', caption)
    
    # Ensure it ends with proper punctuation
    if not caption.endswith(('.', '!', '?')):
        caption += '.'
    
    return caption

def sample_with_temperature(decoder, features, temperature=1.0, max_len=20):
    """Sample captions with temperature for randomness"""
    start_token = vocab.word2idx['<start>']
    end_token = vocab.word2idx['<end>']
    
    sampled_ids = [start_token]
    
    # Initialize hidden state
    h = torch.zeros(decoder.num_layers, 1, decoder.hidden_size).to(device)
    c = torch.zeros(decoder.num_layers, 1, decoder.hidden_size).to(device)
    hidden = (h, c)
    
    # First input is the image features
    inputs = features.unsqueeze(1)
    
    for i in range(max_len):
        # Forward pass
        hiddens, hidden = decoder.lstm(inputs, hidden)
        outputs = decoder.linear(hiddens.squeeze(1))
        
        # Apply temperature
        if temperature > 0:
            outputs = outputs / temperature
            probs = F.softmax(outputs, dim=1)
            # Sample from the distribution
            predicted = torch.multinomial(probs, 1)
        else:
            # Greedy (temperature = 0)
            _, predicted = outputs.max(1)
        
        sampled_ids.append(predicted.item())
        
        # Stop if end token is generated
        if predicted.item() == end_token:
            break
            
        # Next input is the predicted word
        inputs = decoder.embed(predicted)
        inputs = inputs.unsqueeze(1)
    
    return sampled_ids

def beam_search_with_temperature(decoder, features, vocab, beam_size=3, temperature=1.0, max_seq_length=20):
    """Beam search with temperature for diversity"""
    start_token = vocab.word2idx['<start>']
    end_token = vocab.word2idx['<end>']
    
    beams = [([start_token], 0.0)]
    
    for step in range(max_seq_length):
        candidates = []
        
        for seq, score in beams:
            if seq[-1] == end_token:
                candidates.append((seq, score))
                continue
                
            with torch.no_grad():
                # Prepare input
                if len(seq) == 1:
                    inputs = features.unsqueeze(1)
                else:
                    input_word = torch.tensor([seq[-1]]).to(device)
                    inputs = decoder.embed(input_word).unsqueeze(1)
                
                # Get predictions
                hiddens, _ = decoder.lstm(inputs)
                outputs = decoder.linear(hiddens.squeeze(1))
                
                # Apply temperature
                if temperature > 0:
                    outputs = outputs / temperature
                    probs = F.softmax(outputs, dim=1)
                    # Get top k probabilities and indices
                    top_probs, top_indices = probs.topk(beam_size, dim=1)
                else:
                    probs = F.softmax(outputs, dim=1)
                    top_probs, top_indices = probs.topk(beam_size, dim=1)
                
                # Expand beams
                for i in range(beam_size):
                    word_id = top_indices[0][i].item()
                    word_prob = top_probs[0][i].item()
                    
                    new_seq = seq + [word_id]
                    new_score = score + math.log(word_prob + 1e-10)
                    candidates.append((new_seq, new_score))
        
        # Keep top beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        
        # Stop if all beams ended
        if all(beam[0][-1] == end_token for beam in beams):
            break
    
    return beams[0][0]

# Different caption generation strategies
def generate_caption_deterministic(image, encoder, decoder, vocab):
    """Deterministic caption - same every time for same image"""
    image_tensor = transform(image).unsqueeze(0).to(device)
    feature = encoder(image_tensor)
    
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption_words.append(word)
    
    caption = ' '.join(caption_words)
    return caption

def generate_caption_random(image, encoder, decoder, vocab, temperature=1.0):
    """Random caption - different every time"""
    image_tensor = transform(image).unsqueeze(0).to(device)
    feature = encoder(image_tensor)
    
    sampled_ids = sample_with_temperature(decoder, feature, temperature=temperature)
    
    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption_words.append(word)
    
    caption = ' '.join(caption_words)
    return caption

def generate_caption_beam_random(image, encoder, decoder, vocab, beam_size=3, temperature=1.0):
    """Beam search with randomness"""
    image_tensor = transform(image).unsqueeze(0).to(device)
    feature = encoder(image_tensor)
    
    sampled_ids = beam_search_with_temperature(decoder, feature, vocab, 
                                             beam_size=beam_size, temperature=temperature)
    
    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption_words.append(word)
    
    caption = ' '.join(caption_words)
    return caption

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")

# Generation mode selection
st.sidebar.subheader("Generation Mode")
generation_mode = st.sidebar.selectbox(
    "Select generation style:",
    ["Deterministic (Same every time)", "Random (Different every time)", "Beam Search with Randomness"],
    help="Choose how captions are generated"
)

# Parameters based on mode
if generation_mode == "Deterministic (Same every time)":
    use_beam_search = st.sidebar.checkbox("Use Beam Search", value=False)
    beam_size = st.sidebar.slider("Beam Size", 1, 5, 3) if use_beam_search else 1
    
elif generation_mode == "Random (Different every time)":
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                   help="Higher = more random, Lower = more deterministic")
    
elif generation_mode == "Beam Search with Randomness":
    beam_size = st.sidebar.slider("Beam Size", 1, 5, 3)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.1,
                                   help="Higher = more random variations in beam search")

# Display device info
st.sidebar.subheader("System Info")
st.sidebar.write(f"Device: {'GPU 🔥' if torch.cuda.is_available() else 'CPU ⚡'}")

# Upload section
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Store the current image in session state to detect changes
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'generated_captions' not in st.session_state:
    st.session_state.generated_captions = []

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Check if image changed
    if st.session_state.current_image != uploaded_file.name:
        st.session_state.current_image = uploaded_file.name
        st.session_state.generated_captions = []
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Show current configuration
    st.info(f"**Mode:** {generation_mode}")
    if "Random" in generation_mode:
        st.info(f"**Temperature:** {temperature}")
    if "Beam" in generation_mode:
        st.info(f"**Beam Size:** {beam_size}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Generate Single Caption", use_container_width=True):
            with st.spinner("Generating caption..."):
                try:
                    if generation_mode == "Deterministic (Same every time)":
                        caption = generate_caption_deterministic(image, encoder, decoder, vocab)
                    elif generation_mode == "Random (Different every time)":
                        caption = generate_caption_random(image, encoder, decoder, vocab, temperature)
                    else:  # Beam Search with Randomness
                        caption = generate_caption_beam_random(image, encoder, decoder, vocab, beam_size, temperature)
                    
                    final_caption = post_process_caption(caption)
                    
                    if final_caption != "No caption generated":
                        st.session_state.generated_captions.append(final_caption)
                        st.success("✨ Caption Generated!")
                        st.markdown(f'<div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid #4CAF50;">'
                                   f'<h3 style="margin: 0; color: #333;">{final_caption}</h3>'
                                   f'</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.error("❌ No caption was generated.")
                        
                except Exception as e:
                    st.error(f"Error generating caption: {str(e)}")
    
    with col2:
        if st.button("🎲 Generate Multiple Captions", use_container_width=True):
            with st.spinner("Generating multiple captions..."):
                try:
                    captions = []
                    for i in range(3):  # Generate 3 different captions
                        if generation_mode == "Deterministic (Same every time)":
                            caption = generate_caption_deterministic(image, encoder, decoder, vocab)
                        elif generation_mode == "Random (Different every time)":
                            caption = generate_caption_random(image, encoder, decoder, vocab, temperature)
                        else:  # Beam Search with Randomness
                            caption = generate_caption_beam_random(image, encoder, decoder, vocab, beam_size, temperature)
                        
                        final_caption = post_process_caption(caption)
                        if final_caption != "No caption generated":
                            captions.append(final_caption)
                            st.session_state.generated_captions.append(final_caption)
                    
                    if captions:
                        st.success(f"✨ Generated {len(captions)} different captions!")
                        for i, caption in enumerate(captions, 1):
                            st.markdown(f'**Caption {i}:** {caption}')
                    else:
                        st.error("❌ No captions were generated.")
                        
                except Exception as e:
                    st.error(f"Error generating captions: {str(e)}")
    
    # Show caption history for current image
    if st.session_state.generated_captions:
        st.subheader("📝 Caption History for This Image")
        unique_captions = list(dict.fromkeys(st.session_state.generated_captions))  # Remove duplicates while preserving order
        
        for i, caption in enumerate(unique_captions, 1):
            st.write(f"{i}. {caption}")
        
        if st.button("Clear History", type="secondary"):
            st.session_state.generated_captions = []
            st.rerun()

# Explanation of different modes
with st.expander("ℹ️ About Generation Modes"):
    st.markdown("""
    **Deterministic Mode:**
    - Same caption every time for the same image
    - Uses greedy search or deterministic beam search
    - Good for consistent results
    
    **Random Mode:**
    - Different caption every time for the same image
    - Uses temperature sampling to introduce randomness
    - Higher temperature = more creative/random captions
    - Lower temperature = more predictable captions
    
    **Beam Search with Randomness:**
    - Combines beam search with temperature sampling
    - Explores multiple possibilities with some randomness
    - Good balance of quality and diversity
    """)

# Temperature explanation
with st.expander("🎯 Understanding Temperature"):
    st.markdown("""
    **Temperature controls randomness:**
    
    - **Low temperature (0.1-0.5):** More deterministic, focused on high-probability words
    - **Medium temperature (0.5-1.0):** Balanced randomness and coherence  
    - **High temperature (1.0-2.0):** More creative and diverse, but potentially less coherent
    
    **Example for same image:**
    - Low temp: "a black dog is running in the grass"
    - Medium temp: "a dark colored dog runs through a green field" 
    - High temp: "canine sprints across lush lawn chasing a ball"
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Image Captioning App • Built with PyTorch and Streamlit"
    "</div>",
    unsafe_allow_html=True
)











# import streamlit as st
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from PIL import Image
# import pickle
# from model import EncoderCNN, DecoderRNN
# from build_vocab import Vocabulary
# import os
# import math
# import language_tool_python

# st.set_page_config(page_title="Image Captioning", layout="centered")
# st.title("🖼️ Advanced Image Captioning with PyTorch")

# VOCAB_PATH = "data/vocab.pkl"
# ENCODER_PATH = "models/encoder-5-3000.pkl"
# DECODER_PATH = "models/decoder-5-3000.pkl"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @st.cache_resource
# def load_vocab():
#     with open(VOCAB_PATH, "rb") as f:
#         vocab = pickle.load(f)
#     return vocab

# # Load models
# @st.cache_resource
# def load_model():
#     vocab = load_vocab()
#     encoder = EncoderCNN(embed_size=256).eval().to(device)
#     decoder = DecoderRNN(
#         embed_size=256,
#         hidden_size=512,
#         vocab_size=len(vocab),
#         num_layers=1
#     ).eval().to(device)

#     encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
#     decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    
#     return encoder, decoder, vocab

# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406),
#                          (0.229, 0.224, 0.225))
# ])

# encoder, decoder, vocab = load_model()

# def post_process_caption(caption):
#     """Post-process caption for better readability"""
#     if not caption:
#         return "No caption generated"
    
#     # Capitalize first letter
#     caption = caption.capitalize()
    
#     # Fix common issues
#     caption = caption.replace(' a a ', ' a ').replace(' an a ', ' an ')
#     caption = caption.replace(' .', '.').replace(' ,', ',')
    
#     # Remove duplicate words
#     words = caption.split()
#     if len(words) > 1:
#         cleaned_words = [words[0]]
#         for i in range(1, len(words)):
#             if words[i] != words[i-1]:
#                 cleaned_words.append(words[i])
#         caption = ' '.join(cleaned_words)
    
#     # Ensure it ends with proper punctuation
#     if not caption.endswith(('.', '!', '?')):
#         caption += '.'
    
#     return caption

# def grammar_check(caption):
#     """Basic grammar checking"""
#     try:
#         tool = language_tool_python.LanguageTool('en-US')
#         matches = tool.check(caption)
#         corrected = language_tool_python.utils.correct(caption, matches)
#         tool.close()
#         return corrected
#     except Exception as e:
#         st.sidebar.warning(f"Grammar check skipped: {e}")
#         return caption

# # Fixed and improved beam search implementation
# def beam_search(decoder, features, vocab, beam_size=3, max_seq_length=20):
#     """Reliable beam search for caption generation"""
    
#     start_token = vocab.word2idx['<start>']
#     end_token = vocab.word2idx['<end>']
    
#     # Initialize beams with (sequence, score)
#     beams = [([start_token], 0.0)]
    
#     for step in range(max_seq_length):
#         candidates = []
        
#         for seq, score in beams:
#             # If sequence ended, keep it as is
#             if seq[-1] == end_token:
#                 candidates.append((seq, score))
#                 continue
                
#             # Prepare the sequence for the decoder
#             with torch.no_grad():
#                 # For simplicity, we'll use the decoder's sample method in a loop
#                 # This is less efficient but more reliable
#                 if len(seq) == 1:
#                     # First word - use features directly
#                     input_seq = torch.tensor(seq).unsqueeze(0).to(device)
#                     # We need to run the decoder manually for the first step
#                     embeddings = decoder.embed(input_seq)
#                     hiddens, hidden_state = decoder.lstm(embeddings)
#                     outputs = decoder.linear(hiddens.squeeze(1))
#                 else:
#                     # For sequences longer than 1, we need to run through the decoder
#                     # This is a simplified approach
#                     input_seq = torch.tensor(seq).unsqueeze(0).to(device)
#                     # We'll use a temporary approach: run the decoder on the sequence
#                     temp_output = decoder.forward_simple(features, input_seq)
#                     outputs = temp_output[:, -1, :]
                
#                 # Get probabilities
#                 probs = F.softmax(outputs, dim=1)
#                 top_probs, top_indices = probs.topk(beam_size, dim=1)
                
#                 # Expand each beam
#                 for i in range(beam_size):
#                     word_id = top_indices[0][i].item()
#                     word_prob = top_probs[0][i].item()
                    
#                     new_seq = seq + [word_id]
#                     new_score = score + math.log(word_prob + 1e-10)  # Avoid log(0)
                    
#                     candidates.append((new_seq, new_score))
        
#         # Sort candidates by score and keep top beam_size
#         candidates.sort(key=lambda x: x[1], reverse=True)
#         beams = candidates[:beam_size]
        
#         # Check if all beams have ended
#         if all(beam[0][-1] == end_token for beam in beams):
#             break
    
#     # Return the best sequence
#     return beams[0][0]

# # Alternative: Simple greedy search with temperature
# def sample_with_temperature(decoder, features, temperature=1.0):
#     """Sample with temperature for diversity"""
#     sampled_ids = decoder.sample(features)
#     return sampled_ids[0].cpu().numpy()

# # Enhanced caption generation
# def generate_enhanced_caption(image, encoder, decoder, vocab, use_beam_search=True, beam_size=3, use_grammar_check=True):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     feature = encoder(image_tensor)
    
#     try:
#         if use_beam_search:
#             sampled_ids = beam_search(decoder, feature, vocab, beam_size=beam_size)
#         else:
#             sampled_ids = decoder.sample(feature)
#             sampled_ids = sampled_ids[0].cpu().numpy()
#     except Exception as e:
#         st.warning(f"Beam search failed: {e}. Falling back to greedy search.")
#         sampled_ids = decoder.sample(feature)
#         sampled_ids = sampled_ids[0].cpu().numpy()
    
#     # Convert to caption
#     caption_words = []
#     for word_id in sampled_ids:
#         word = vocab.idx2word[word_id]
#         if word == '<end>':
#             break
#         if word != '<start>':
#             caption_words.append(word)
    
#     caption = ' '.join(caption_words)
    
#     # Post-processing
#     caption = post_process_caption(caption)
    
#     # Optional grammar check
#     if use_grammar_check:
#         caption = grammar_check(caption)
    
#     return caption

# # Simple caption generation (fallback)
# def generate_caption(image, encoder, decoder, vocab):
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     feature = encoder(image_tensor)
#     sampled_ids = decoder.sample(feature)
#     sampled_ids = sampled_ids[0].cpu().numpy()
    
#     caption = []
#     for word_id in sampled_ids:
#         word = vocab.idx2word[word_id]
#         if word == '<end>':
#             break
#         if word != '<start>':
#             caption.append(word)
#     caption = ' '.join(caption)
#     return post_process_caption(caption)

# # Sidebar for configuration
# st.sidebar.title("⚙️ Configuration")

# # Model options
# st.sidebar.subheader("Model Settings")
# use_beam_search = st.sidebar.checkbox("Use Beam Search", value=True, 
#                                      help="Better quality but slightly slower")
# beam_size = st.sidebar.slider("Beam Size", min_value=1, max_value=5, value=3, 
#                              help="Higher values = better quality but slower")

# # Post-processing options
# st.sidebar.subheader("Post-processing")
# use_grammar_check = st.sidebar.checkbox("Grammar Check", value=True,
#                                        help="Apply grammar correction to captions")
# use_advanced_processing = st.sidebar.checkbox("Advanced Processing", value=True,
#                                             help="Apply advanced caption cleaning")

# # Upload section
# st.subheader("📤 Upload Image")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_container_width=True)
    
#     # Show current configuration
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.info(f"**Search:** {'Beam' if use_beam_search else 'Greedy'}")
#     with col2:
#         st.info(f"**Beam Size:** {beam_size}")
#     with col3:
#         st.info(f"**Grammar Check:** {'On' if use_grammar_check else 'Off'}")

#     # Generate caption button
#     if st.button("🚀 Generate Caption", type="primary"):
#         with st.spinner("Generating caption... This may take a few seconds."):
#             try:
#                 if use_advanced_processing:
#                     caption = generate_enhanced_caption(
#                         image, encoder, decoder, vocab, 
#                         use_beam_search=use_beam_search, 
#                         beam_size=beam_size,
#                         use_grammar_check=use_grammar_check
#                     )
#                 else:
#                     caption = generate_caption(image, encoder, decoder, vocab)
                
#                 # Display result
#                 st.success("✨ Caption Generated!")
#                 st.markdown(f'<div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid #4CAF50;"><h3 style="margin: 0; color: #333;">{caption}</h3></div>', 
#                            unsafe_allow_html=True)
                
#             except Exception as e:
#                 st.error(f"Error generating caption: {str(e)}")
#                 st.info("Try using simpler settings or a different image.")

# # Information section
# st.sidebar.markdown("---")
# st.sidebar.subheader("ℹ️ About")
# st.sidebar.info(
#     """
#     **Beam Search**: Considers multiple word sequences for better quality captions.
    
#     **Grammar Check**: Uses language tool to correct grammar and spelling.
    
#     **Advanced Processing**: Applies caption cleaning and formatting.
#     """
# )

# # Performance tips
# with st.expander("💡 Performance Tips"):
#     st.markdown("""
#     - **For best results**: Use beam search with beam size 3-5
#     - **For faster results**: Disable beam search and grammar check
#     - **Image quality**: Use clear, well-lit images for better captions
#     - **Common objects**: Works best with everyday objects and scenes
#     """)

# # Add some sample images for testing
# st.sidebar.markdown("---")
# st.sidebar.subheader("🖼️ Sample Images")
# st.sidebar.markdown("Try with images containing:")
# st.sidebar.markdown("- People and animals\n- Common objects\n- Outdoor scenes\n- Clear backgrounds")
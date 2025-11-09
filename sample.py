import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import PIL.Image as Image
from models import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary
import argparse
import heapq

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def beam_search(decoder, features, vocab, beam_size=5, max_seq_length=20):
    """Beam search for caption generation"""
    
    # Start with the start token
    start_token = vocab['<start>']
    end_token = vocab['<end>']
    
    # Initialize beam with start token and score 0
    beams = [([start_token], 0.0, decoder.get_initial_state(features))]
    
    for _ in range(max_seq_length):
        new_beams = []
        
        for seq, score, hidden_state in beams:
            # If the sequence already ended, just carry it forward
            if seq[-1] == end_token:
                new_beams.append((seq, score, hidden_state))
                continue
                
            # Prepare input
            input_word = torch.tensor([seq[-1]]).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs, hidden_state = decoder(features, input_word, hidden_state)
                word_logprobs = F.log_softmax(outputs, dim=1)
                topk_logprobs, topk_indices = word_logprobs.topk(beam_size, dim=1)
            
            # Expand beam
            for i in range(beam_size):
                word_idx = topk_indices[0][i].item()
                word_logprob = topk_logprobs[0][i].item()
                new_seq = seq + [word_idx]
                new_score = score + word_logprob
                new_beams.append((new_seq, new_score, hidden_state))
        
        # Keep only top beam_size sequences
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Check if all beams ended
        if all(beam[0][-1] == end_token for beam in beams):
            break
    
    # Return the best sequence
    best_sequence = beams[0][0]
    return best_sequence

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate caption using beam search
    feature = encoder(image_tensor)
    
    if args.use_beam_search:
        sampled_ids = beam_search(decoder, feature, vocab, beam_size=args.beam_size)
    else:
        # Fallback to greedy search
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab[str(word_id)]
        sampled_caption.append(word)
        if word == '<end>':
            break
    
    sentence = ' '.join(sampled_caption[1:-1])  # Remove <start> and <end> tokens
    
    # Print out the image and the generated caption
    print(sentence)
    
    # Save the generated caption
    with open(args.result_path, 'w') as f:
        f.write(sentence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.json', help='path for vocabulary wrapper')
    parser.add_argument('--result_path', type=str, default='result.txt', help='path for saving the result')
    parser.add_argument('--use_beam_search', action='store_true', help='use beam search instead of greedy')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size for beam search')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)



























# import torch
# import matplotlib.pyplot as plt
# import numpy as np 
# import argparse
# import pickle 
# import os
# from torchvision import transforms 
# from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
# from PIL import Image


# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def load_image(image_path, transform=None):
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize([224, 224], Image.LANCZOS)
    
#     if transform is not None:
#         image = transform(image).unsqueeze(0)
    
#     return image

# def main(args):
#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(), 
#         transforms.Normalize((0.485, 0.456, 0.406), 
#                              (0.229, 0.224, 0.225))])
    
#     # Load vocabulary wrapper
#     with open(args.vocab_path, 'rb') as f:
#         vocab = pickle.load(f)

#     # Build models
#     encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
#     encoder = encoder.to(device)
#     decoder = decoder.to(device)

#     # Load the trained model parameters
#     encoder.load_state_dict(torch.load(args.encoder_path))
#     decoder.load_state_dict(torch.load(args.decoder_path))

#     # Prepare an image
#     image = load_image(args.image, transform)
#     image_tensor = image.to(device)
    
#     # Generate an caption from the image
#     feature = encoder(image_tensor)
#     sampled_ids = decoder.sample(feature)
#     sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
#     # Convert word_ids to words
#     sampled_caption = []
#     for word_id in sampled_ids:
#         word = vocab.idx2word[word_id]
#         sampled_caption.append(word)
#         if word == '<end>':
#             break
#     sentence = ' '.join(sampled_caption)
    
#     # Print out the image and the generated caption
#     print (sentence)
#     image = Image.open(args.image)
#     plt.imshow(np.asarray(image))
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
#     parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
#     parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
#     parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
#     # Model parameters (should be same as paramters in train.py)
#     parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
#     parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
#     parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
#     args = parser.parse_args()
#     main(args)

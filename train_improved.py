import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import numpy as np
from model_improved import EncoderCNN, DecoderRNN
from dataset import get_loader
import math

def train_improved():
    # Hyperparameters
    embed_size = 512
    hidden_size = 1024
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 20
    train_cnn = False
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loader
    data_loader, dataset = get_loader(batch_size=32, num_workers=2)
    vocab_size = len(dataset.vocab)
    
    # Models
    encoder = EncoderCNN(embed_size, train_cnn).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word2idx['<pad>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    # Training
    total_step = len(data_loader)
    
    for epoch in range(num_epochs):
        # Scheduled sampling probability
        teacher_forcing_ratio = max(0.5, 1 - epoch / num_epochs)
        
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
            
            # Forward pass
            features = encoder(images)
            outputs, alphas = decoder(features, captions, lengths)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Add doubly stochastic attention regularization
            loss += 1.0 * ((1. - torch.cat(alphas, 1).sum(dim=1)) ** 2).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            nn.utils.clip_grad_norm_(encoder.linear.parameters(), 5.0)
            
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f'models/encoder_improved_{epoch+1}.pkl')
            torch.save(decoder.state_dict(), f'models/decoder_improved_{epoch+1}.pkl')

if __name__ == '__main__':
    train_improved()
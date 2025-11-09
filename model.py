import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use ResNet-152 (matches the pre-trained model)
        resnet = models.resnet152(pretrained=True)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze all ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Add a linear layer to transform features to embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        # Initialize hidden state
        h = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
        c = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
        hidden = (h, c)
        
        for i in range(20):  # maximum sampling length
            hiddens, hidden = self.lstm(inputs, hidden)  
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids




# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torch.nn.utils.rnn import pack_padded_sequence

# class Attention(nn.Module):
#     """Attention Network"""
#     def __init__(self, encoder_dim, decoder_dim, attention_dim):
#         super(Attention, self).__init__()
#         self.encoder_att = nn.Linear(encoder_dim, attention_dim)
#         self.decoder_att = nn.Linear(decoder_dim, attention_dim)
#         self.full_att = nn.Linear(attention_dim, 1)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, encoder_out, decoder_hidden):
#         att1 = self.encoder_att(encoder_out)
#         att2 = self.decoder_att(decoder_hidden)
#         att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
#         alpha = self.softmax(att)
#         attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
#         return attention_weighted_encoding, alpha

# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size, train_cnn=False):
#         super(EncoderCNN, self).__init__()
#         self.train_cnn = train_cnn
#         # Use ResNet-101 for better features
#         resnet = models.resnet101(pretrained=True)
        
#         # Remove the last fully connected layer
#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules)
        
#         # Fine-tuning
#         for param in self.resnet.parameters():
#             param.requires_grad = train_cnn
            
#         # ResNet feature size is 2048
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
#         self.fine_tune(fine_tune=train_cnn)

#     def forward(self, images):
#         features = self.resnet(images)
#         features = self.adaptive_pool(features)
#         features = features.permute(0, 2, 3, 1)
#         features = features.view(features.size(0), -1, features.size(-1))
#         return features

#     def fine_tune(self, fine_tune=True):
#         for p in self.resnet.parameters():
#             p.requires_grad = False
#         # Only fine-tune convolutional blocks 2-4
#         for c in list(self.resnet.children())[5:]:
#             for p in c.parameters():
#                 p.requires_grad = fine_tune

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers, attention_dim=512, dropout=0.5):
#         super(DecoderRNN, self).__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
        
#         # Attention
#         self.attention = Attention(embed_size, hidden_size, attention_dim)
        
#         # Embedding
#         self.embed = nn.Embedding(vocab_size, embed_size)
        
#         # LSTM with dropout
#         self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
#         # Linear layers
#         self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, vocab_size)
        
#         self.dropout = nn.Dropout(p=dropout)
#         self.init_weights()

#     def init_weights(self):
#         self.embed.weight.data.uniform_(-0.1, 0.1)
#         self.fc2.bias.data.fill_(0)
#         self.fc2.weight.data.uniform_(-0.1, 0.1)

#     def forward(self, features, captions, lengths):
#         embeddings = self.embed(captions)
#         embeddings = self.dropout(embeddings)
        
#         # Initialize hidden state
#         batch_size = features.size(0)
#         h, c = self.init_hidden(batch_size)
        
#         # Sort sequences by length
#         lengths, sort_ind = lengths.sort(dim=0, descending=True)
#         features = features[sort_ind]
#         embeddings = embeddings[sort_ind]
        
#         predictions = []
#         alphas = []
        
#         for t in range(max(lengths)):
#             batch_size_t = sum([l > t for l in lengths])
#             if batch_size_t == 0:
#                 break
                
#             attention_weighted_encoding, alpha = self.attention(
#                 features[:batch_size_t], h[:batch_size_t].squeeze(0))
            
#             lstm_input = torch.cat([embeddings[:batch_size_t, t], attention_weighted_encoding], dim=1)
#             lstm_input = lstm_input.unsqueeze(1)
            
#             h = h[:, :batch_size_t]
#             c = c[:, :batch_size_t]
            
#             _, (h, c) = self.lstm(lstm_input, (h, c))
#             output = self.fc1(torch.cat([h.squeeze(0), attention_weighted_encoding], dim=1))
#             pred = self.fc2(self.dropout(output))
            
#             predictions.append(pred)
#             alphas.append(alpha)
            
#         return torch.cat(predictions, 0), alphas

#     def sample(self, features, max_len=20):
#         with torch.no_grad():
#             features = features.squeeze(0)
#             batch_size = features.size(0)
            
#             h, c = self.init_hidden(1)
#             words = torch.tensor([[self.vocab_size - 3]]).to(features.device)  # <start>
            
#             captions = []
#             alphas = []
            
#             for i in range(max_len):
#                 embeddings = self.embed(words).squeeze(1)
                
#                 attention_weighted_encoding, alpha = self.attention(features, h.squeeze(0))
#                 alphas.append(alpha.cpu().detach().numpy())
                
#                 lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
#                 lstm_input = lstm_input.unsqueeze(1)
                
#                 _, (h, c) = self.lstm(lstm_input, (h, c))
#                 output = self.fc1(torch.cat([h.squeeze(0), attention_weighted_encoding], dim=1))
#                 pred = self.fc2(self.dropout(output))
                
#                 predicted = pred.argmax(1)
#                 words = predicted.unsqueeze(0)
                
#                 captions.append(predicted.item())
#                 if predicted.item() == self.vocab_size - 2:  # <end>
#                     break
                    
#             return torch.tensor(captions), alphas

#     def init_hidden(self, batch_size):
#         h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
#         c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
#         return h.to(self.embed.weight.device), c.to(self.embed.weight.device)


# # import torch
# # import torch.nn as nn
# # import torchvision.models as models
# # from torch.nn.utils.rnn import pack_padded_sequence


# # class EncoderCNN(nn.Module):
# #     def __init__(self, embed_size):
# #         """Load the pretrained ResNet-152 and replace top fc layer."""
# #         super(EncoderCNN, self).__init__()
# #         resnet = models.resnet152(pretrained=True)
# #         modules = list(resnet.children())[:-1]      # delete the last fc layer.
# #         self.resnet = nn.Sequential(*modules)
# #         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
# #         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
# #     def forward(self, images):
# #         """Extract feature vectors from input images."""
# #         with torch.no_grad():
# #             features = self.resnet(images)
# #         features = features.reshape(features.size(0), -1)
# #         features = self.bn(self.linear(features))
# #         return features


# # class DecoderRNN(nn.Module):
# #     def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
# #         """Set the hyper-parameters and build the layers."""
# #         super(DecoderRNN, self).__init__()
# #         self.embed = nn.Embedding(vocab_size, embed_size)
# #         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
# #         self.linear = nn.Linear(hidden_size, vocab_size)
# #         self.max_seg_length = max_seq_length
        
# #     def forward(self, features, captions, lengths):
# #         """Decode image feature vectors and generates captions."""
# #         embeddings = self.embed(captions)
# #         embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
# #         packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
# #         hiddens, _ = self.lstm(packed)
# #         outputs = self.linear(hiddens[0])
# #         return outputs
    
# #     def sample(self, features, states=None):
# #         """Generate captions for given image features using greedy search."""
# #         sampled_ids = []
# #         inputs = features.unsqueeze(1)
# #         for i in range(self.max_seg_length):
# #             hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
# #             outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
# #             _, predicted = outputs.max(1)                        # predicted: (batch_size)
# #             sampled_ids.append(predicted)
# #             inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
# #             inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
# #         sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
# #         return sampled_ids
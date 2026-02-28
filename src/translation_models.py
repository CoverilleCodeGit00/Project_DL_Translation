import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Encodeur Bi-LSTM
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        
        # Projection pour adapter la dimension bidirectionnelle au decodeur unidirectionnel
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        
        # outputs = [batch size, src len, enc hid dim * 2]
        # hidden/cell = [2, batch size, enc hid dim] (car 1 couche bidirectionnelle)
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # Concatenation des etats caches forward et backward (indices 0 et 1)
        # hidden_cat = [batch size, enc hid dim * 2]
        hidden_cat = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
        cell_cat = torch.cat((cell[0,:,:], cell[1,:,:]), dim=1)
        
        # Creation du Vecteur de Contexte (Le goulot d'etranglement)
        # s_hidden = [1, batch size, dec hid dim]
        s_hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)
        s_cell = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)
        
        return s_hidden, s_cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # Decodeur LSTM unidirectionnel SANS attention
        self.rnn = nn.LSTM(emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size] (1 seul token a la fois)
        input = input.unsqueeze(1) # [batch size, 1]
        
        embedded = self.dropout(self.embedding(input)) # [batch size, 1, emb dim]
        
        # Le decodeur ne recoit que l'etat precedent (qui est initialement le vecteur de contexte global)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [batch size, 1, dec hid dim]
        
        prediction = self.fc_out(output.squeeze(1)) # [batch size, output dim]
        
        return prediction, hidden, cell


class Seq2SeqVanilla(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor pour stocker les predictions du decodeur
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # L'encodeur traite la phrase source et produit les vecteurs de contexte
        hidden, cell = self.encoder(src)
        
        # La premiere entree du decodeur est le token <sos> (Start Of Sequence)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Le decodeur genere le token suivant en se basant sur le contexte et l'entree precedente
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            
            # Decision d'utiliser le Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            # Si teacher forcing, on utilise le vrai token cible, sinon on utilise la prediction du modele
            input = trg[:, t] if teacher_force else top1
            
        return outputs
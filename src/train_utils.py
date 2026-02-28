import torch
import torch.nn as nn
import time
import math
import sacrebleu
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm

# Telechargement silencieux des ressources NLTK pour METEOR
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def train_epoch(model, dataloader, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    """
    Entraine le modele sur une epoque complete avec affichage dynamique de la loss.
    """
    model.train()
    epoch_loss = 0
    
    # Creation de la barre de progression tqdm
    iterator = tqdm(dataloader, desc="Entrainement", leave=False)
    
    for i, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        
        output = model(src, trg, teacher_forcing_ratio)
        
        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Mise a jour dynamique de la loss dans la barre tqdm a chaque batch
        iterator.set_postfix(loss=f"{loss.item():.4f}")
        
    return epoch_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion):
    """
    Evalue le modele avec affichage dynamique de la loss.
    """
    model.eval()
    epoch_loss = 0
    
    iterator = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            output = model(src, trg, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            # Mise a jour dynamique
            iterator.set_postfix(loss=f"{loss.item():.4f}")
            
    return epoch_loss / len(dataloader)


def calculate_metrics(model, iterator, sp_en, device):
    """
    Calcule BLEU, chrF et METEOR sur le set de validation pour le suivi qualitatif.
    """
    model.eval()
    targets = []
    predictions = []
    
    with torch.no_grad():
        for src, trg in iterator:
            # src et trg sont deja sur le bon device via le dataloader
            output = model(src, trg, teacher_forcing_ratio=0)
            
            # Recuperation de l'ID avec la plus haute probabilite
            output_ids = output.argmax(dim=-1)
            
            # Decodage du batch
            for i in range(src.shape[0]):
                pred_tokens = output_ids[i].tolist()
                trg_tokens = trg[i].tolist()
                
                # Nettoyage des tokens speciaux (pad:0, unk:1, bos:2, eos:3)
                pred_tokens = [t for t in pred_tokens if t not in [0, 2, 3]]
                trg_tokens = [t for t in trg_tokens if t not in [0, 2, 3]]
                
                # Conversion des IDs en texte via SentencePiece
                pred_text = sp_en.decode_ids(pred_tokens)
                trg_text = sp_en.decode_ids(trg_tokens)
                
                predictions.append(pred_text)
                targets.append([trg_text]) # sacrebleu attend une liste de references par prediction
                
    # Calcul avec sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, targets).score
    chrf = sacrebleu.corpus_chrf(predictions, targets).score
    
    # METEOR (phrase par phrase via NLTK)
    meteor_scores = []
    for p, t in zip(predictions, targets):
        # NLTK attend des listes de mots pour METEOR
        meteor_scores.append(meteor_score([t[0].split()], p.split()))
    
    meteor = (sum(meteor_scores) / len(meteor_scores)) * 100 if meteor_scores else 0.0
    
    return bleu, chrf, meteor


def train_model(model, train_iterator, valid_iterator, optimizer, criterion, sp_en, n_epochs, clip, save_path, device):
    """
    Boucle principale d'entrainement.
    Gere la sauvegarde du meilleur modele (Best Model Checkpoint).
    """
    best_valid_loss = float('inf')
    
    print(f"Demarrage de l'entrainement pour {n_epochs} epoques...")
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # 1. Entrainement (mise a jour des poids)
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, clip, teacher_forcing_ratio=0.5)
        
        # 2. Evaluation (mesure de la generalisation)
        valid_loss = evaluate_epoch(model, valid_iterator, criterion)
        
        # 3. Calcul des metriques linguistiques
        bleu, chrf, meteor = calculate_metrics(model, valid_iterator, sp_en, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # 4. Sauvegarde anti-overfitting
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            saved_msg = "=> Checkpoint sauvegarde !"
        else:
            saved_msg = ""
            
        print(f"Epoque: {epoch+1:02} | Temps: {epoch_mins}m {epoch_secs:.0f}s {saved_msg}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(min(train_loss, 100)):7.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(min(valid_loss, 100)):7.3f}")
        print(f"\tMetriques Val -> BLEU: {bleu:.2f} | chrF: {chrf:.2f} | METEOR: {meteor:.2f}\n")

    print(f"Entrainement termine. Meilleur modele sauvegarde sous : {save_path}")
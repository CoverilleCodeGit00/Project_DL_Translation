import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

def train_sentencepiece_models(train_csv_path, vocab_size=8000, model_dir="models"):
    """
    Entraine les modeles SentencePiece (BPE) uniquement sur le set de Train.
    Exporte des modeles separes pour le francais (source) et l'anglais (cible).
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Chargement des donnees d'entrainement depuis {train_csv_path}")
    df_train = pd.read_csv(train_csv_path)

    # Fichiers temporaires purs textes pour SentencePiece
    fr_text_file = os.path.join(model_dir, "temp_train_fr.txt")
    en_text_file = os.path.join(model_dir, "temp_train_en.txt")

    # Ecriture des textes bruts dans les fichiers temporaires
    with open(fr_text_file, 'w', encoding='utf-8') as f_fr, \
         open(en_text_file, 'w', encoding='utf-8') as f_en:
        for _, row in df_train.iterrows():
            f_fr.write(str(row['text_fr']) + '\n')
            f_en.write(str(row['text_en']) + '\n')

    print("Entrainement du modele SentencePiece FR...")
    spm.SentencePieceTrainer.train(
        input=fr_text_file,
        model_prefix=os.path.join(model_dir, "spm_fr"),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>"
    )

    print("Entrainement du modele SentencePiece EN...")
    spm.SentencePieceTrainer.train(
        input=en_text_file,
        model_prefix=os.path.join(model_dir, "spm_en"),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>"
    )

    # Nettoyage des fichiers temporaires
    os.remove(fr_text_file)
    os.remove(en_text_file)
    print("Entrainement SentencePiece termine avec succes. Fichiers sauvegardes dans :", model_dir)


class TranslationDataset(Dataset):
    """
    Classe Dataset PyTorch avec possibilite de sous-echantillonnage (sample_frac).
    """
    def __init__(self, csv_file, spm_fr_path, spm_en_path, sample_frac=1.0):
        super().__init__()
        # Chargement du dataframe complet
        df = pd.read_csv(csv_file)
        
        # Sous-echantillonnage dynamique si on veut juste tester (ex: 0.2 pour 20%)
        if sample_frac < 1.0:
            # random_state=42 garantit qu'on prend toujours les memes 20% a chaque lancement
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            print(f"Dataset reduit a {sample_frac*100}% : {len(df)} phrases chargees.")
        else:
            print(f"Dataset complet : {len(df)} phrases chargees.")
            
        self.data = df
        
        # Initialisation des tokenizers
        self.sp_fr = spm.SentencePieceProcessor()
        self.sp_fr.load(spm_fr_path)
        
        self.sp_en = spm.SentencePieceProcessor()
        self.sp_en.load(spm_en_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fr_sentence = str(self.data.iloc[idx]['text_fr'])
        en_sentence = str(self.data.iloc[idx]['text_en'])

        fr_ids = [self.sp_fr.bos_id()] + self.sp_fr.encode_as_ids(fr_sentence) + [self.sp_fr.eos_id()]
        en_ids = [self.sp_en.bos_id()] + self.sp_en.encode_as_ids(en_sentence) + [self.sp_en.eos_id()]

        return torch.tensor(fr_ids, dtype=torch.long), torch.tensor(en_ids, dtype=torch.long)


def collate_fn_translation(batch, pad_idx=0):
    """
    Fonction pour aligner (pad) les sequences d'un batch a la meme longueur.
    A passer dans l'argument 'collate_fn' du DataLoader PyTorch.
    """
    src_batch, trg_batch = [], []
    
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
        
    # Padding des sequences pour qu'elles aient la taille de la sequence la plus longue du batch
    src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=pad_idx, batch_first=True)
    
    return src_batch, trg_batch
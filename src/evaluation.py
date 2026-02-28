import pandas as pd
import re
import jiwer
from pathlib import Path

class ASREvaluator:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        # On supprime les lignes ou la prediction a echoue completement
        self.df = self.df.dropna(subset=['text_fr', 'text_pred']).copy()
        
        # Transformation de nettoyage standard ASR
        self.normalization_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ExpandCommonEnglishContractions() # Optionnel mais utile
        ])

    def _normalize(self, text):
        """Nettoie le texte pour une comparaison honnete (sans ponctuation, en minuscules)"""
        try:
            return self.normalization_transform(str(text))
        except:
            return ""

    def compute_global_metrics(self):
        """Calcule le WER et le CER globaux apres normalisation"""
        print("Normalisation du texte en cours...")
        self.df['ref_norm'] = self.df['text_fr'].apply(self._normalize)
        self.df['hyp_norm'] = self.df['text_pred'].apply(self._normalize)
        
        # On retire les cas ou la reference est totalement vide apres nettoyage
        valid_df = self.df[self.df['ref_norm'].str.len() > 0].copy()
        
        references = valid_df['ref_norm'].tolist()
        hypotheses = valid_df['hyp_norm'].tolist()
        
        # Calcul standard
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        
        # Calcul detaille ligne par ligne pour l'analyse
        valid_df['wer'] = valid_df.apply(lambda row: jiwer.wer(row['ref_norm'], row['hyp_norm']), axis=1)
        self.df = valid_df
        
        return wer, cer

    def get_worst_segments(self, top_n=10):
        """Renvoie les segments avec le pire WER pour comprendre les failles du modele"""
        return self.df.sort_values(by='wer', ascending=False).head(top_n)

    def analyze_error_categories(self):
        """Analyse la performance sur des sous-categories complexes (Bruit, Chiffres, Noms propres)"""
        
        # 1. Bruits / Didascalies (Detection de parentheses ou crochets)
        mask_noise = self.df['text_fr'].str.contains(r'\(|\[|\)|\]', regex=True)
        
        # 2. Chiffres (Detection de digits)
        mask_numbers = self.df['text_fr'].str.contains(r'\d', regex=True)
        
        # 3. Noms Propres (Detection grossiere de majuscules en milieu de phrase)
        # Ex: "je suis allé à Paris"
        mask_proper = self.df['text_fr'].str.contains(r'\b[a-z]+ \b[A-Z][a-z]+', regex=True)
        
        categories = {
            'Didascalies & Bruits (Ex: Applaudissements)': mask_noise,
            'Segments avec Chiffres': mask_numbers,
            'Segments avec Noms Propres (Majuscules)': mask_proper,
            'Standard (Aucun des trois)': ~(mask_noise | mask_numbers | mask_proper)
        }
        
        results = []
        for cat_name, mask in categories.items():
            sub_df = self.df[mask]
            if len(sub_df) > 0:
                refs = sub_df['ref_norm'].tolist()
                hyps = sub_df['hyp_norm'].tolist()
                cat_wer = jiwer.wer(refs, hyps)
                results.append({'Categorie': cat_name, 'Nb Segments': len(sub_df), 'WER': cat_wer})
                
        return pd.DataFrame(results)
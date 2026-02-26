import pandas as pd
from pathlib import Path
import whisper
import tarfile
import os
from tqdm import tqdm
import torch

class ASRPredictor:
    def __init__(self, root_dir: Path, model_size="small"):
        self.root_dir = Path(root_dir)
        self.temp_dir = self.root_dir / "data" / "temp"
        self.processed_dir = self.root_dir / "data" / "processed"
        
        # Detection automatique du GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initialisation ASR ---")
        print(f"Device detecte : {self.device.upper()}")
        print(f"Chargement du modele Whisper ({model_size}). Cela peut prendre un instant...")
        
        # Chargement du modele (small est un bon compromis vitesse/qualite pour commencer)
        self.model = whisper.load_model(model_size, device=self.device)
        print("✅ Modele charge et pret !")

    def _format_timestamp(self, seconds: float) -> str:
        """Convertit des secondes en format SRT: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def extract_single_flac(self, archive_name: str, video_id: str) -> Path:
        """Extrait un seul fichier FLAC complet pour le test de l'Etape 1"""
        tar_path = self.root_dir / "data" / "raw" / archive_name
        flac_filename = f"{video_id}.flac"
        
        print(f"Extraction temporaire de {flac_filename}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith(flac_filename)), None)
            if not member:
                raise FileNotFoundError(f"Audio {flac_filename} introuvable dans l'archive.")
            
            tar.extract(member, path=self.temp_dir)
            return self.temp_dir / member.name

    def transcribe_full_video_to_srt(self, audio_path: Path, output_srt_path: Path, language="fr"):
        """Etape 1: Transcrit un audio complet et genere un vrai fichier SRT"""
        print(f"Transcription de la video complete : {audio_path.name}")
        
        # L'ASR ecoute toute la video
        result = self.model.transcribe(str(audio_path), language=language)
        
        # Generation du fichier SRT
        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                # Format standard SRT
                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{text}\n\n")
                
        print(f"✅ Fichier SRT genere avec succes : {output_srt_path}")
        return result["text"]

    def transcribe_segments_to_csv(self, csv_path: Path, split: str, limit: int = None):
        """Etape 2: Transcrit nos petits segments decoupes et cree un CSV comparatif"""
        print(f"--- Transcription des segments ({split.upper()}) ---")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Index CSV introuvable : {csv_path}")
            
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)
            
        segments_dir = self.processed_dir / "segments" / split
        predictions = []
        
        # Boucle sur les segments
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference ASR"):
            seg_id = row['segment_id']
            audio_path = segments_dir / f"{seg_id}.wav"
            
            if audio_path.exists():
                # On desactive les logs internes de Whisper (fp16 warning) pour un affichage propre
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Inference !
                    # On met fp16=False si on est sur CPU pour eviter des avertissements
                    fp16_mode = True if self.device == "cuda" else False
                    result = self.model.transcribe(str(audio_path), language="fr", fp16=fp16_mode)
                    
                predicted_text = result["text"].strip()
            else:
                predicted_text = None
                
            predictions.append(predicted_text)
            
        # On ajoute la colonne predite
        df['text_pred'] = predictions
        
        # Sauvegarde dans un nouveau CSV special "Resultats"
        output_csv = self.processed_dir / f"{split}_asr_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Resultats sauvegardes dans : {output_csv}")
        
        return df
import tarfile
import pandas as pd
from pathlib import Path

class MultilingualIndexer:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "data" / "raw"
        self.temp_dir = self.root_dir / "data" / "temp"
        self.processed_dir = self.root_dir / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_metadata_files(self, archives: list):
        # Creation du dossier temporaire
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        for archive_name in archives:
            tar_path = self.raw_dir / archive_name
            if not tar_path.exists():
                print(f"Info : Archive {archive_name} non trouvee (ignoree).")
                continue
            
            print(f"Extraction rapide de {archive_name}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                # Filtre pour extraire uniquement les fichiers texte/config, pas l'audio lourd
                members = [m for m in tar.getmembers() 
                           if m.name.endswith(('.yaml', '.fr', '.en', '.es', '.pt', '.txt')) 
                           or 'segments' in m.name]
                tar.extractall(path=self.temp_dir, members=members)

    def _load_aligned_lang(self, folder: str, split: str, lang_ext: str, col_name: str):
        """
        Charge une paire (segments, texte) pour une langue donnee et renvoie un DataFrame.
        Essentiel pour garantir l'alignement par ID.
        """
        base_path = self.temp_dir / folder / "data" / split / "txt"
        segments_file = base_path / "segments"
        text_file = base_path / f"{split}.{lang_ext}"

        # 1. Verification d'existence
        if not segments_file.exists() or not text_file.exists():
            return None

        # 2. Lecture des IDs (depuis le fichier segments specifique)
        df_seg = pd.read_csv(segments_file, sep=r'\s+', header=None, usecols=[0], names=['segment_id'])
        
        # 3. Lecture du Texte ligne par ligne
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        
        # 4. Verification de synchronisation
        if len(df_seg) != len(lines):
            print(f"ATTENTION ({folder}): Decalage detecte ! {len(df_seg)} segments vs {len(lines)} lignes.")
            min_len = min(len(df_seg), len(lines))
            df_seg = df_seg.iloc[:min_len]
            lines = lines[:min_len]

        # 5. Creation du DataFrame aligne
        df_seg[col_name] = lines
        return df_seg

    def _read_kaldi_segments_full(self, file_path):
        """Lit le fichier segments complet pour la structure temporelle FR"""
        if not file_path.exists():
            return None
        df = pd.read_csv(file_path, sep=r'\s+', header=None, 
                         names=['segment_id', 'video_id', 'start', 'end'], 
                         dtype={'start': float, 'end': float})
        return df

    def create_master_index(self, split: str = 'train'):
        print(f"Construction de l'Index Maitre (Mode Aligne ID) pour : {split}")
        
        # --- ETAPE 1 : La Reference FR (Structure + Texte) ---
        base_fr_path = self.temp_dir / "fr-fr" / "data" / split / "txt"
        segments_fr = base_fr_path / "segments"
        
        if not segments_fr.exists():
            raise FileNotFoundError(f"Segments FR introuvables : {segments_fr}")

        # 1. Structure (Temps + Video ID)
        df_master = self._read_kaldi_segments_full(segments_fr)
        print(f"Total Segments FR : {len(df_master)}")

        # 2. Chemins Relatifs (Portables pour Kaggle/Colab)
        # L'utilisation de Path("data") assure un chemin relatif propre
        df_master['wav_path'] = df_master['video_id'].apply(
            lambda x: str(Path("data") / "temp" / "fr-fr" / "data" / split / "wav" / f"{x}.wav")
        )
        df_master['vtt_path'] = df_master['video_id'].apply(
            lambda x: str(Path("data") / "temp" / "fr-fr" / "data" / split / "vtt" / f"{x}.vtt")
        )

        # 3. Texte FR (Alignement ID)
        df_text_fr = self._load_aligned_lang("fr-fr", split, "fr", "text_fr")
        if df_text_fr is not None:
            df_master = pd.merge(df_master, df_text_fr, on='segment_id', how='left')
        else:
            raise ValueError("Critique : Impossible de charger le texte FR !")

        # --- ETAPE 2 : Les Traductions (Alignement ID) ---
        langs = [('fr-en', 'en'), ('fr-es', 'es'), ('fr-pt', 'pt')]
        
        for folder, lang in langs:
            col_name = f'text_{lang}'
            df_trans = self._load_aligned_lang(folder, split, lang, col_name)
            
            if df_trans is not None:
                # Merge LEFT : On garde tous les segments FR.
                df_master = pd.merge(df_master, df_trans, on='segment_id', how='left')
            else:
                df_master[col_name] = None

        # --- ETAPE 3 : Finalisation ---
        output_file = self.processed_dir / "master_metadata.csv"
        df_master = df_master.rename(columns={'start': 'start_time', 'end': 'end_time'})
        
        df_master.to_csv(output_file, index=False)
        print(f"Succes ! CSV Maitre genere : {output_file} ({len(df_master)} lignes)")
        
        return df_master

    def create_specialized_datasets(self, master_df=None):
        """
        Divise le Master Index en sous-datasets specialises (ASR vs NMT).
        """
        print("\n--- Generation des Datasets Specialises ---")
        
        # Chargement depuis le disque si non fourni
        if master_df is None:
            master_path = self.processed_dir / "master_metadata.csv"
            if not master_path.exists():
                print("Erreur: master_metadata.csv introuvable. Executez create_master_index d'abord.")
                return
            master_df = pd.read_csv(master_path)

        # 1. DATASET ASR (Transcription pure)
        # On garde tout ce qui a de l'audio et du texte FR
        df_asr = master_df[['segment_id', 'video_id', 'start_time', 'end_time', 'wav_path', 'text_fr']].dropna(subset=['text_fr', 'wav_path'])
        
        path_asr = self.processed_dir / "train_asr_fr.csv"
        df_asr.to_csv(path_asr, index=False)
        print(f"✅ [ASR] Dataset FR genere : {path_asr} ({len(df_asr)} segments)")
        print(f"   -> Pour Fine-tuning Whisper")

        # 2. DATASETS NMT (Traduction)
        langs = ['en', 'es', 'pt']
        
        for lang in langs:
            col_target = f'text_{lang}'
            
            # Verification si la colonne existe
            if col_target not in master_df.columns:
                print(f"Info: Colonne {col_target} manquante. Pas de dataset NMT genere.")
                continue
                
            cols = ['segment_id', 'video_id', 'start_time', 'end_time', 'wav_path', 'text_fr', col_target]
            
            # On ne garde que les paires completes (FR + TRAD)
            df_nmt = master_df[cols].dropna(subset=['text_fr', col_target])
            
            path_nmt = self.processed_dir / f"train_nmt_fr_{lang}.csv"
            df_nmt.to_csv(path_nmt, index=False)
            print(f"✅ [NMT] Dataset Parallele FR-{lang.upper()} genere : {path_nmt} ({len(df_nmt)} paires)")

    def calculate_coverage_stats(self, df: pd.DataFrame):
        total = len(df)
        print(f"\n--- Statistiques de Couverture (N={total}) ---")
        for lang in ['en', 'es', 'pt']:
            col = f'text_{lang}'
            if col in df.columns:
                count = df[col].notna().sum()
                pct = (count / total) * 100
                print(f"Traduction {lang.upper()}: {count} ({pct:.2f}%)")
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
        """Etape 1: Extraction des fichiers texte (txt) pour chaque split et langue."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        for archive_name in archives:
            tar_path = self.raw_dir / archive_name
            if not tar_path.exists():
                print(f"Info : Archive {archive_name} non trouvée.")
                continue
            print(f"Extraction des métadonnées de {archive_name}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() 
                           if m.name.endswith(('.yaml', '.fr', '.en', '.es', '.pt', '.txt')) 
                           or 'segments' in m.name]
                tar.extractall(path=self.temp_dir, members=members)
    
    def present_raw_dataset(self, splits=['train', 'valid', 'test']):
        """Etape 2: Présentation du dataset brut (fichiers extraits)"""
        print("=== PRÉSENTATION DU DATASET (Fichiers Bruts extraits) ===")
        
        for split in splits:
            print(f"\n📁 SPLIT : {split.upper()}")
            
            split_name = split
            base_fr_path = self.temp_dir / "fr-fr" / "data" / split_name / "txt"
            
            if not base_fr_path.exists():
                print("  -> Dossier introuvable.")
                continue
                
            # 1. Analyse des Segments (Vidéos)
            segments_file = base_fr_path / "segments"
            if segments_file.exists():
                df_seg = pd.read_csv(segments_file, sep=r'\s+', header=None, usecols=[0, 1], names=['segment_id', 'video_id'])
                nb_segments = len(df_seg)
                nb_videos = df_seg['video_id'].nunique()
                print(f"  🎬 Vidéos uniques (Source FR) : {nb_videos}")
                print(f"  ⏱️ Segments audio totaux    : {nb_segments}")
            
            # 2. Analyse des Fichiers Textes (Lignes)
            print("  📝 Contenu des fichiers textes :")
            
            # Fichier Français
            fr_file = base_fr_path / f"{split_name}.fr"
            if fr_file.exists():
                with open(fr_file, 'r', encoding='utf-8') as f:
                    print(f"     - Français (.fr) : {sum(1 for _ in f)} lignes")
            
            # Fichiers de Traduction
            for lang in ['en', 'es', 'pt']:
                trans_dir = self.temp_dir / f"fr-{lang}" / "data" / split_name / "txt"
                trans_file = trans_dir / f"{split_name}.{lang}"
                if trans_file.exists():
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        print(f"     - Traduction {lang.upper()} (.{lang}) : {sum(1 for _ in f)} lignes")

    def _load_aligned_lang(self, folder: str, split: str, lang_ext: str, col_name: str):
        base_path = self.temp_dir / folder / "data" / split / "txt"
        segments_file = base_path / "segments"
        text_file = base_path / f"{split}.{lang_ext}"

        if not segments_file.exists() or not text_file.exists():
            return None

        df_seg = pd.read_csv(segments_file, sep=r'\s+', header=None, usecols=[0], names=['segment_id'])
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        
        if len(df_seg) != len(lines):
            min_len = min(len(df_seg), len(lines))
            df_seg = df_seg.iloc[:min_len]
            lines = lines[:min_len]

        df_seg[col_name] = lines
        return df_seg

    def _read_kaldi_segments_full(self, file_path):
        if not file_path.exists():
            return None
        return pd.read_csv(file_path, sep=r'\s+', header=None, 
                           names=['segment_id', 'video_id', 'start', 'end'], 
                           dtype={'start': float, 'end': float})

    def create_master_index(self, split: str):
        """Etape 3: Construction d'un csv index general pour un split."""
        print(f"Construction de l'Index Maitre pour : {split}")
        
        base_fr_path = self.temp_dir / "fr-fr" / "data" / split / "txt"
        segments_fr = base_fr_path / "segments"
        
        if not segments_fr.exists():
            if split == 'valid':
                return self.create_master_index(split='dev')
            else:
                return None

        df_master = self._read_kaldi_segments_full(segments_fr)

        # Chemins virtuels FLAC
        df_master['wav_path'] = df_master['video_id'].apply(
            lambda x: str(Path("data") / "temp" / "fr-fr" / "data" / split / "wav" / f"{x}.flac")
        )

        # Textes FR
        df_text_fr = self._load_aligned_lang("fr-fr", split, "fr", "text_fr")
        if df_text_fr is not None:
            df_master = pd.merge(df_master, df_text_fr, on='segment_id', how='left')
        
        # Traductions
        for folder, lang in [('fr-en', 'en'), ('fr-es', 'es'), ('fr-pt', 'pt')]:
            df_trans = self._load_aligned_lang(folder, split, lang, f'text_{lang}')
            if df_trans is not None:
                df_master = pd.merge(df_master, df_trans, on='segment_id', how='left')
            else:
                df_master[f'text_{lang}'] = None

        save_split_name = 'valid' if split == 'dev' else split
        output_file = self.processed_dir / f"master_{save_split_name}.csv"
        df_master = df_master.rename(columns={'start': 'start_time', 'end': 'end_time'})
        df_master.to_csv(output_file, index=False)
        return df_master

    def create_specialized_datasets(self, split: str):
        """Etape 3b: Séparation des fichiers pour chaque langue."""
        master_path = self.processed_dir / f"master_{split}.csv"
        if not master_path.exists(): return

        master_df = pd.read_csv(master_path)
        
        # ASR (FR uniquement)
        df_asr = master_df[['segment_id', 'video_id', 'start_time', 'end_time', 'wav_path', 'text_fr']].dropna(subset=['text_fr'])
        df_asr.to_csv(self.processed_dir / f"{split}_asr_fr.csv", index=False)

        # NMT (Paires de langues)
        for lang in ['en', 'es', 'pt']:
            col = f'text_{lang}'
            if col in master_df.columns:
                df_nmt = master_df[['segment_id', 'text_fr', col]].dropna()
                if not df_nmt.empty:
                    df_nmt.to_csv(self.processed_dir / f"{split}_nmt_fr_{lang}.csv", index=False)

    def analyze_dataset(self, splits=['train', 'valid', 'test']):
        """Etape 4: Analyse poussee du dataset (Volumes, Heures, Traductions)."""
        stats = []
        for split in splits:
            master_path = self.processed_dir / f"master_{split}.csv"
            if not master_path.exists(): continue
            
            df = pd.read_csv(master_path)
            
            # Calcul des durees
            if 'start_time' in df.columns and 'end_time' in df.columns:
                df['duration'] = df['end_time'] - df['start_time']
                total_hours = df['duration'].sum() / 3600
                avg_duration = df['duration'].mean()
            else:
                total_hours = 0
                avg_duration = 0

            total_fr = df['text_fr'].notna().sum()
            videos = df['video_id'].nunique()
            
            row = {
                'Split': split, 
                'Videos': videos, 
                'Segments (FR)': total_fr,
                'Heures Audio': round(total_hours, 2),
                'Duree moy/seg (s)': round(avg_duration, 2)
            }
            
            # Analyse de la deperdition des traductions par rapport au FR
            for lang in ['en', 'es', 'pt']:
                if f'text_{lang}' in df.columns:
                    count = df[f'text_{lang}'].notna().sum()
                    row[f'Paires FR-{lang.upper()}'] = count
                    row[f'% {lang.upper()} (vs FR)'] = round((count / total_fr) * 100, 2) if total_fr > 0 else 0
            stats.append(row)
            
        return pd.DataFrame(stats)
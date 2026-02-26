import tarfile
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import os
import shutil
from tqdm import tqdm

class AudioProcessor:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "data" / "raw"
        self.temp_dir = self.root_dir / "data" / "temp"
        self.processed_dir = self.root_dir / "data" / "processed" / "segments"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("❌ ERREUR: FFmpeg non détecté. Requis pour pydub.")

    def extract_and_segment(self, split: str, archive_name: str = "mtedx_fr.tgz", limit: int = None):
        """Etape 5: Extraction des sons pour un split donné (depuis les FLAC)."""
        csv_path = self.root_dir / "data" / "processed" / f"{split}_asr_fr.csv"
        if not csv_path.exists():
            print(f"⚠️ Index introuvable pour {split}. Avez-vous exécuté le Notebook 1 ?")
            return
        
        df = pd.read_csv(csv_path)
        if limit: df = df.head(limit)

        grouped = df.groupby('video_id')
        tar_path = self.raw_dir / archive_name
        output_split_dir = self.processed_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Extraction Audio pour {split.upper()} ({len(grouped)} vidéos) ---")

        with tarfile.open(tar_path, "r:gz") as tar:
            for video_id, group in tqdm(grouped, desc=f"Segmentation {split}"):
                flac_filename = f"{video_id}.flac"
                member = next((m for m in tar.getmembers() if m.name.endswith(flac_filename)), None)
                
                if not member: continue

                tar.extract(member, path=self.temp_dir)
                temp_flac_path = self.temp_dir / member.name
                
                try:
                    full_audio = AudioSegment.from_file(temp_flac_path, format="flac")
                    full_audio = full_audio.set_frame_rate(16000).set_channels(1)
                    
                    for _, row in group.iterrows():
                        start_ms, end_ms = int(row['start_time'] * 1000), int(row['end_time'] * 1000)
                        segment_audio = full_audio[start_ms:min(end_ms, len(full_audio))]
                        out_name = output_split_dir / f"{row['segment_id']}.wav"
                        segment_audio.export(out_name, format="wav", codec="pcm_s16le")
                        
                finally:
                    if temp_flac_path.exists():
                        os.remove(temp_flac_path)
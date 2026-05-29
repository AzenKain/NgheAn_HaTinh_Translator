import os
import torch
from pathlib import Path
from transformers import pipeline

class ASRService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.getenv("ASR_MODEL_PATH", "build/onnx-web/merged/whisper-lora-nghe-tinh")
        self.model_path = str(Path(model_path).resolve())
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ASR Model path not found at: {self.model_path}. Make sure to run export script first.")
            
        print(f"[ASR] Loading Whisper model from: {self.model_path}...")
        
        # Detect device
        if torch.cuda.is_available():
            device = 0
            torch_dtype = torch.float16
            print("[ASR] CUDA available, using GPU (fp16)")
        else:
            device = -1
            torch_dtype = torch.float32
            print("[ASR] CUDA not available, using CPU (fp32)")
            
        # Initialize pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_path,
            chunk_length_s=30,
            device=device,
            torch_dtype=torch_dtype
        )
        print("[ASR] Model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        if self.pipe is None:
            self.load_model()
            
        print(f"[ASR] Transcribing audio: {audio_path}")
        try:
            import soundfile as sf
            import numpy as np
            
            # Read audio data natively using soundfile
            speech_array, sampling_rate = sf.read(audio_path)
            
            # Convert stereo to mono
            if len(speech_array.shape) > 1:
                speech_array = speech_array.mean(axis=1)
                
            # Resample to 16000Hz if necessary using simple numpy linear interpolation
            target_sr = 16000
            if sampling_rate != target_sr:
                print(f"[ASR] Resampling audio from {sampling_rate}Hz to {target_sr}Hz...")
                length_new = int(round(len(speech_array) * target_sr / sampling_rate))
                speech_array = np.interp(
                    np.linspace(0, len(speech_array) - 1, length_new),
                    np.arange(len(speech_array)),
                    speech_array
                ).astype(np.float32)
                sampling_rate = target_sr
            
            # Pass raw array to pipeline (bypasses ffmpeg file loader)
            result = self.pipe(
                {"raw": speech_array, "sampling_rate": sampling_rate},
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=True,
                generate_kwargs={
                    "language": "vi",
                    "task": "transcribe",
                    "forced_decoder_ids": None,
                    "repetition_penalty": 1.15,
                    "no_repeat_ngram_size": 5
                }
            )
            return result.get("text", "")
        except Exception as e:
            print(f"[ASR] Transcription failed: {e}")
            raise e

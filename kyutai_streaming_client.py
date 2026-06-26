"""
kyutai_streaming_client.py
==========================

A lightweight local STT client for Kyutai's sherpa-small model.
No API key required. Runs fully offline.

Usage:
    from kyutai_streaming_client import KyutaiStreamingClient

    client = KyutaiStreamingClient(model="sherpa-small", language="fr")
    text = client.transcribe_bytes(audio_bytes)
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download

import torch
import torchaudio


class KyutaiStreamingClient:
    def __init__(self, model="sherpa-small", language="fr"):
        self.model_name = model
        self.language = language

        # Local cache directory
        self.cache_dir = Path.home() / ".napoleon" / "kyutai"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.model, self.decoder = self._load_model()

    # ----------------------------------------------------------------------
    # Model loader
    # ----------------------------------------------------------------------
    def _load_model(self):
        """
        Downloads sherpa-small from Hugging Face if missing,
        then loads it into PyTorch.
        """

        repo = "k2-fsa/sherpa-onnx-zipformer-small-fr"
        files = {
            "encoder": "encoder-epoch-99-avg-1.onnx",
            "decoder": "decoder-epoch-99-avg-1.onnx",
            "joiner": "joiner-epoch-99-avg-1.onnx",
            "tokens": "tokens.txt",
        }

        local_paths = {}

        for key, filename in files.items():
            local_paths[key] = hf_hub_download(
                repo_id=repo,
                filename=filename,
                cache_dir=self.cache_dir,
            )

        import sherpa_onnx

        model = sherpa_onnx.OfflineRecognizer(
            tokens=local_paths["tokens"],
            encoder=local_paths["encoder"],
            decoder=local_paths["decoder"],
            joiner=local_paths["joiner"],
            num_threads=4,
            provider="cpu",
        )

        return model, None

    # ----------------------------------------------------------------------
    # Audio transcription
    # ----------------------------------------------------------------------
    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Takes raw audio bytes (wav/mp3/m4a) and returns a transcript string.
        """

        # Write to temp file
        tmp_path = self.cache_dir / "tmp_input.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # Load audio
        audio, sr = sf.read(tmp_path)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16 kHz
        if sr != 16000:
            audio = torchaudio.functional.resample(
                torch.tensor(audio), sr, 16000
            ).numpy()

        # Run STT
        text = self.model.decode(audio)

        return text.strip()

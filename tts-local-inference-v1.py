"""
Standalone GPT-SoVITS TTS inference (English only)
=================================================================
Extracted directly from GPT_SoVITS/inference_webui.py.

This code uses reverse engineer to only extract required 
functions / packages for inference given fine tuned models.
"""

import os
import sys

# GPT-SoVITS's internal modules (cnhubert.py, process_ckpt.py, etc.) use
# bare imports like `import utils` expecting GPT_SoVITS/ itself to be on
# sys.path — not just the repo root. The original webui scripts get this
# for free by living inside GPT_SoVITS/; since this script lives one level
# up, we add that path manually.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPT_SOVITS_DIR = os.path.join(SCRIPT_DIR, "GPT_SoVITS")
if GPT_SOVITS_DIR not in sys.path:
    sys.path.insert(0, GPT_SOVITS_DIR)

import pathlib

# Workaround: GPT checkpoints trained on Linux/Mac may have pathlib.PosixPath
# objects pickled inside (e.g. a stored config path). Windows can't
# instantiate PosixPath directly. Temporarily alias it to WindowsPath so
# torch.load can deserialize the checkpoint without erroring.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

import nltk
nltk.download("cmudict", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Standard Packages
import json
import torch
import librosa
import numpy as np
from time import time as ttime
import soundfile as sf
import torchaudio



# Project Packages
from GPT_SoVITS.sv import SV
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

######################################################################################
# GLOBAL VARIABLES
######################################################################################

# Punctuation marks treated as sentence terminators
SENTENCE_TERMINATORS = {
    "。", ",", ".", "?", "!", "~", ":", "—", "…"
}
# GPT model's fixed semantic-token generation rate (tokens per second). Used as a hard stop during GPT sampling: early_stop_num = GPT_TOKEN_RATE_HZ * max_sec
GPT_TOKEN_RATE_HZ = 50


######################################################################################
# RESAMPLING HELPER
# Lifted directly from inference_webui.py — caches torchaudio Resample
# transforms by (sr0, sr1) key to avoid rebuilding them on every call.
######################################################################################
_resample_transform_cache = {}

def resample_audio(audio_tensor: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    '''
    Resamples audio_tensor from source_sr to target_sr, caching the
    torchaudio Resample transform so it's only built once per rate pair.

    Parameters
        audio_tensor: (1, N) or (N,) float32 waveform tensor.
        source_sr: Source sample rate in Hz.
        target_sr: Target sample rate in Hz.

    Returns
        Resampled waveform tensor on CPU.
    '''
    key = (source_sr, target_sr)
    if key not in _resample_transform_cache:
        _resample_transform_cache[key] = torchaudio.transforms.Resample(source_sr, target_sr)
    return _resample_transform_cache[key](audio_tensor)


######################################################################################
# REFERENCE SPECTROGRAM EXTRACTOR
# Lifted from get_spepc() in inference_webui.py, CPU-only, v2Pro path always
# active (always returns the 16kHz audio tensor needed for sv_emb computation).
######################################################################################
def get_reference_spectrogram(
    reference_wav_path: str,
    hps,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Loads the reference audio, computes its linear spectrogram (used by the
    SoVITS decoder as a timbre/acoustic reference), and returns the audio
    resampled to 16kHz (needed by the SV model for speaker-embedding).

    Parameters
        reference_wav_path: Path to the reference audio file.
        hps: SoVITS hyperparameters (provides filter_length, hop_length,
             win_length, sampling_rate).

    Returns
        Tuple of:
            reference_spectrogram: (1, freq_bins, T) float32 tensor.
            audio_16k: (1, N) float32 tensor at 16kHz, for SV embedding.
    '''
    model_sr = int(hps.data.sampling_rate)

    #audio, file_sr = torchaudio.load(reference_wav_path)
    audio_np, file_sr = sf.read(reference_wav_path, dtype="float32")
    audio = torch.from_numpy(audio_np).T  # soundfile returns (samples, channels), torch expects (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # mono → (1, samples)

    # Stereo → mono
    if audio.shape[0] == 2:
        audio = audio.mean(0).unsqueeze(0)

    # Resample to model's native rate if needed
    if file_sr != model_sr:
        audio = resample_audio(audio, file_sr, model_sr)

    # Clip gain if needed (prevents spectrogram saturation)
    peak = audio.abs().max()
    if peak > 1:
        audio /= min(2, peak)

    # Compute linear spectrogram at model's native sample rate
    reference_spectrogram = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    ).float()

    # Resample to 16kHz for SV embedding (v2Pro always needs this)
    audio_16k = resample_audio(audio, model_sr, 16000).float()

    return reference_spectrogram, audio_16k


######################################################################################
# SV MODEL LOADER
######################################################################################

######################################################################################
# TEXT FRONTEND HELPERS
######################################################################################
def get_phones_and_bert(
    text: str,
    language: str,
    version: str = "v2",
) -> tuple[list, torch.Tensor, str]:
    '''
    Converts text into phoneme IDs and BERT features for the GPT model.

    For English (language == "en"), the BERT features are a zero tensor —
    BERT conditioning is only meaningful for Chinese/Japanese/Korean. The
    zero tensor acts as a neutral placeholder of the correct shape.

    Parameters
        text: Input sentence, already normalized and punctuated.
        language: Language code — "en" for English.
        version: Model version string — always "v2" here.

    Returns
        Tuple of:
            phones: list of phoneme token IDs.
            bert: (1024, N_phones) float32 BERT feature tensor.
            norm_text: cleaned/normalized text string (for logging).
    '''
    language_for_cleaner = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language_for_cleaner, version)
    phones = cleaned_text_to_sequence(phones, version)

    # English: return a zero BERT tensor of the correct shape
    bert = torch.zeros(1024, len(phones), dtype=torch.float32)

    return phones, bert, norm_text


######################################################################################
# MODEL LOADERS HELPER FUNCTIONS
######################################################################################
class DictToAttrRecursive(dict):
    '''
    Wraps a nested dict so its keys are also accessible as attributes
    (e.g. hps.data.sampling_rate instead of hps["data"]["sampling_rate"]).
    This is exactly how the original script exposes the SoVITS checkpoint's
    embedded hyperparameters (`hps`).
    '''
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)


def load_sv_model(sv_checkpoint_path: str):
    '''
    Loads the ERes2NetV2 speaker-verification model used by v2Pro to compute
    speaker embeddings (sv_emb) passed into vq_model.decode().

    The SV model is a fixed, pretrained model — it is NOT fine-tuned per
    voice. It extracts a speaker identity embedding from a reference audio
    clip, conditioning the decoder to stay in that speaker's timbre even
    when generating new content.

    Parameters
        sv_checkpoint_path: Path to pretrained_eres2netv2w24s4ep4.ckpt,
                            found under GPT_SoVITS/pretrained_models/sv/.

    Returns
        Loaded SV model instance, in eval mode, on CPU.
    '''
    # SV class uses a hardcoded relative path internally — we override it
    # by pointing sv.sv_path at the actual location before instantiating.
    from GPT_SoVITS import sv as sv_module
    sv_module.sv_path = sv_checkpoint_path

    # SV.__init__ takes (device, is_half) — CPU-only, always float32
    sv_model = SV(device="cpu", is_half=False)
    return sv_model


def load_ssl_model(cnhubert_base_path: str):
    '''
    Loads the Chinese-HuBERT SSL feature extractor. This is the model that
    converts raw reference audio into frame-level semantic features in
    process_reference_audio(). It is not fine-tuned per-voice — it's a
    fixed, general-purpose feature extractor shared across all voices.

    Parameters
        cnhubert_base_path: Path to the pretrained cnhubert checkpoint folder
                             (e.g. "GPT_SoVITS/pretrained_models/chinese-hubert-base").

    Returns
        The loaded SSL model, in eval mode, on CPU.
    '''
    cnhubert.cnhubert_base_path = cnhubert_base_path
    ssl_model = cnhubert.get_model()
    ssl_model = ssl_model.to("cpu")
    ssl_model.eval()
    return ssl_model


def load_sovits_model(sovits_checkpoint_path: str, sovits_config_path: str):
    '''
    Loads your fine-tuned v2 SoVITS checkpoint and its architecture config.

    Your checkpoint is a raw training checkpoint (top-level keys:
    model/iteration/optimizer/learning_rate), not an inference-ready export
    (which would bundle weight/config together) — so the config is loaded
    from a separate JSON file instead of being read out of the checkpoint.

    Parameters
        sovits_checkpoint_path: Path to your fine-tuned SoVITS .pth file.
        sovits_config_path: Path to the matching s2.json architecture config.

    Returns
        Tuple of (vq_model, hps).
    '''
    checkpoint = load_sovits_new(sovits_checkpoint_path)

    with open(sovits_config_path, "r", encoding="utf-8") as f:
        hps = DictToAttrRecursive(json.load(f))

    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = "v2Pro"

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    if "pretrained" not in sovits_checkpoint_path:
        try:
            del vq_model.enc_q
        except AttributeError:
            pass

    vq_model = vq_model.to("cpu")
    vq_model.eval()
    vq_model.load_state_dict(checkpoint["model"], strict=False)

    return vq_model, hps


def load_gpt_model(gpt_checkpoint_path: str, gpt_config_path: str):
    '''
    Loads your fine-tuned GPT (Text2Semantic) checkpoint and its config.

    Your checkpoint is a raw PyTorch Lightning training checkpoint (top-level
    keys: state_dict/hyper_parameters/etc.), not an inference-ready export
    (which would bundle weight/config directly) — so:
      1. The config is loaded from a separate JSON file (extracted from this
         same checkpoint's hyper_parameters at the time it was inspected).
      2. The state_dict keys are stripped of their "model." prefix, since
         Lightning wraps the actual model as self.model inside
         Text2SemanticLightningModule, but the model itself expects
         unprefixed keys (e.g. "bert_proj.weight", not "model.bert_proj.weight").

    Parameters
        gpt_checkpoint_path: Path to your fine-tuned GPT .ckpt file.
        gpt_config_path: Path to the matching s1_config.json architecture config.

    Returns
        Tuple of:
            t2s_model: the loaded GPT model, in eval mode, on CPU.
            max_sec: max seconds of audio the GPT can generate per call.
    '''
    checkpoint = torch.load(gpt_checkpoint_path, map_location="cpu", weights_only=False)

    with open(gpt_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    raw_state_dict = checkpoint["state_dict"]
    model_state_dict = {
        key[len("model."):] if key.startswith("model.") else key: value
        for key, value in raw_state_dict.items()
    }

    max_sec = config["data"]["max_sec"]

    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)

    load_result = t2s_model.model.load_state_dict(model_state_dict, strict=True)
    print("GPT weights loaded:", load_result)

    t2s_model = t2s_model.to("cpu")
    t2s_model.eval()

    return t2s_model, max_sec

######################################################################################
# INFERENCE HELPER FUNCTIONS
######################################################################################
def normalize_text(raw_text: str, language_code: str, log_label: str) -> str:
    '''
    Strips stray newlines and guarantees the text ends with sentence-final
    punctuation. The phonemizer and GPT model expect complete sentences —
    a missing terminal mark can throw off phoneme/prosody alignment.

    Parameters
        raw_text: Input text (reference transcript or target text).
        language_code: Internal language code (e.g. "en", "all_zh") —
                       determines which punctuation mark gets appended.
        log_label: Label used in the debug print (e.g. "Reference text",
                   "Target text") so logs make clear which string this is.

    Returns
        The normalized text string.
    '''
    cleaned_text = raw_text.strip("\n")

    if len(cleaned_text) > 0 and cleaned_text[-1] not in SENTENCE_TERMINATORS:
        cleaned_text += "。" if language_code != "en" else "."

    print(f"{log_label} text (normalized):", cleaned_text)

    return cleaned_text


def split_text_into_chunks(
    text: str,
    method: str = "by_punctuation",
    max_words: int = 50,
    merge_threshold_chars: int = 5,
) -> list[str]:
    '''
    Splits target text into chunks for per-sentence synthesis. Long text is
    split because the GPT model has a max generation length and produces
    more stable, consistent prosody on shorter inputs.

    Two methods:
    "n_words"
        hard cutoff — groups exactly `max_words` words per chunk, regardless of punctuation.
    "by_punctuation"
        Splits on English sentence-ending periods ('.').
        If any resulting sentence still exceeds `max_words`, it is recursively halved by word count until every chunk is within the limit.

    After splitting, every chunk is guaranteed to end with a period (since
    this implementation is English-only), and any chunk shorter than
    `merge_threshold_chars` characters is merged into the previous chunk
    rather than synthesized alone (a 2-3 character fragment produces
    unstable output if synthesized on its own).

    Parameters
        text: The full target text to split.
        method: "n_words" or "by_punctuation".
        max_words: Maximum words allowed in a single chunk. Acts as a hard
                   cap for "n_words" and a safety-net cap for
                   "by_punctuation" (forcing a split if a sentence is too long).
        merge_threshold_chars: Chunks shorter than this (in characters)
                               get merged into the previous chunk.

    Returns
        List of text chunks, each ending in a period, ready for
        per-sentence synthesis.
    '''
    if method not in {"n_words", "by_punctuation"}:
        raise ValueError(f"Unknown split method: {method!r}. Use 'n_words' or 'by_punctuation'.")

    cleaned_text = text.strip("\n")
    words = cleaned_text.split()

    if len(words) == 0:
        return []

    # --- Step 1: produce raw chunks based on the chosen method ---

    if method == "n_words":
        raw_chunks = [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    else:  # "by_punctuation"
        # Split on periods, keeping each sentence's own period attached.
        sentence_chunks = [
            sentence.strip() + "."
            for sentence in cleaned_text.split(".")
            if sentence.strip()
        ]

        raw_chunks = []
        for sentence in sentence_chunks:
            raw_chunks.extend(_halve_until_within_limit(sentence, max_words))

    # --- Step 2: ensure every chunk ends with a period ---
    punctuated_chunks = [
        chunk if chunk.endswith(".") else chunk + "."
        for chunk in raw_chunks
    ]

    # --- Step 3: merge any chunk shorter than merge_threshold_chars into the previous one ---

    merged_chunks = []
    for chunk in punctuated_chunks:
        if len(chunk) < merge_threshold_chars and merged_chunks:
            merged_chunks[-1] = merged_chunks[-1] + " " + chunk
        else:
            merged_chunks.append(chunk)

    return merged_chunks


def _halve_until_within_limit(sentence: str, max_words: int) -> list[str]:
    '''
    Recursively splits a sentence in half (by word count) until every
    resulting piece is at or under max_words. Used as the safety net for
    "by_punctuation" splitting when a single sentence is too long.

    Parameters
        sentence: A single sentence (already ending in a period).
        max_words: The word-count ceiling for each piece.

    Returns
        List of one or more sentence fragments, each within the word limit.
    '''
    words = sentence.rstrip(".").split()

    if len(words) <= max_words:
        return [sentence]

    midpoint = len(words) // 2
    first_half = " ".join(words[:midpoint])
    second_half = " ".join(words[midpoint:])

    return (
        _halve_until_within_limit(first_half + ".", max_words)
        + _halve_until_within_limit(second_half + ".", max_words)
    )


def build_silence_buffer(sample_rate: int, pause_seconds: float) -> torch.Tensor:
    '''
    Builds a buffer of digital silence used both as padding for the SSL
    model (to avoid boundary artifacts at the end of the reference audio)
    and as the audible gap inserted between synthesized sentences.

    Parameters
        sample_rate: The model's native sample rate (samples per second).
        pause_seconds: Desired silence duration, in seconds.

    Returns
        A 1D float32 tensor of zeros, `sample_rate * pause_seconds` samples long.
    '''
    silence_sample_count = int(sample_rate * pause_seconds)
    silence_array = np.zeros(silence_sample_count, dtype=np.float32)
    return torch.from_numpy(silence_array)


def process_reference_audio(
    reference_wav_path: str,
    silence_buffer: torch.Tensor,
    ssl_model,
    vq_model,
) -> torch.Tensor:
    '''
    Loads the reference audio, validates its duration, and runs it through
    the SSL model + VQ codebook to extract the semantic "voice token"
    sequence the GPT model uses to clone the reference speaker.

    This only needs to run once per reference audio — the resulting
    `voice_prompt_tokens` tensor is reused across every sentence synthesized.

    Parameters
        reference_wav_path: Path to the reference audio file (3-10 seconds).
        silence_buffer: Pre-built silence tensor (from build_silence_buffer),
                         appended to the reference waveform before SSL
                         feature extraction to avoid boundary artifacts.
        ssl_model: Loaded SSL feature extractor (from load_ssl_model).
        vq_model: Loaded SoVITS model (from load_sovits_model).

    Returns
        voice_prompt_tokens: (1, T_codes) tensor — semantic token sequence
        representing the reference voice, fed into the GPT model.
    '''
    with torch.no_grad():
        reference_waveform_16k, sample_rate = librosa.load(reference_wav_path, sr=16000)

        if reference_waveform_16k.shape[0] > 160000 or reference_waveform_16k.shape[0] < 48000:
            duration_seconds = reference_waveform_16k.shape[0] / 16000
            raise OSError(
                f"Reference audio must be 3-10 seconds. "
                f"Got {duration_seconds:.1f}s — please use a different clip."
            )

        reference_waveform_16k = torch.from_numpy(reference_waveform_16k)
        padded_waveform = torch.cat([reference_waveform_16k, silence_buffer])

        # SSL model: extracts frame-level semantic features from raw audio.
        # last_hidden_state shape: (1, T, feature_dim) -> transpose -> (1, feature_dim, T)
        ssl_features = ssl_model.model(padded_waveform.unsqueeze(0))["last_hidden_state"].transpose(1, 2)

        # VQ model: quantizes SSL features into discrete semantic tokens.
        # quantized_codes shape: (batch, num_codebooks, T_codes)
        quantized_codes = vq_model.extract_latent(ssl_features)

        # Codebook 0, batch 0 -> the voice's semantic token sequence over time.
        voice_semantic_tokens = quantized_codes[0, 0]

        # Add batch dimension back: (T_codes,) -> (1, T_codes)
        voice_prompt_tokens = voice_semantic_tokens.unsqueeze(0)

    return voice_prompt_tokens


def generate_tts_on_cpu(
    reference_wav_path: str,
    reference_text: str,
    text_to_generate: str,
    ssl_model,
    vq_model,
    t2s_model,
    sv_model,
    hps,
    max_sec: float,
    text_language: str = 'en',
    reference_language: str = 'en',
    parameter_top_k: int = 20,
    parameter_top_p: float = 0.6,
    parameter_temperature: float = 1.0,
    parameter_audio_speed: float = 1.0,
    parameter_pause_seconds: float = 0.3,
    parameter_reuse_gpt_tokens: bool = False,
) -> tuple[int, np.ndarray]:
    '''
    Generates TTS audio using GPT-SoVITS v2Pro, CPU-only, English-only.

    Pipeline per sentence chunk:
      1. Phonemize + BERT-encode reference + target text (BERT is zero for English).
      2. Concatenate reference + target phoneme sequences (few-shot GPT conditioning).
      3. GPT (t2s_model) predicts semantic tokens for the target sentence.
      4. SV model computes a speaker embedding from the reference audio.
      5. SoVITS decoder (vq_model) renders predicted tokens + reference spectrogram
         + speaker embedding into a waveform.
      6. Peak-normalize and append silence gap.
    All sentence chunks are concatenated into one final waveform.

    Parameters
        reference_wav_path: Path to reference audio file (3-10 seconds).
        reference_text: Transcript of the reference audio.
        text_to_generate: Text to synthesize.
        ssl_model: Loaded SSL feature extractor (from load_ssl_model).
        vq_model: Loaded SoVITS v2Pro model (from load_sovits_model).
        t2s_model: Loaded GPT model (from load_gpt_model).
        sv_model: Loaded speaker-verification model (from load_sv_model).
        hps: SoVITS hyperparameters (from load_sovits_model).
        max_sec: Max seconds the GPT can generate per call (from load_gpt_model).
        text_language: Language code for text_to_generate (default "en").
        reference_language: Language code for reference_text (default "en").
        parameter_top_k: GPT top-K sampling.
        parameter_top_p: GPT nucleus sampling.
        parameter_temperature: GPT temperature.
        parameter_audio_speed: Speech rate multiplier.
        parameter_pause_seconds: Silence gap between sentences in seconds.
        parameter_reuse_gpt_tokens: If True, reuse cached GPT tokens per chunk.

    Returns
        Tuple of (sample_rate: int, audio: np.ndarray[int16])
    '''
    gpt_cache = {}

    # --- Validation ---
    if not reference_wav_path:
        raise ValueError("reference_wav_path is required.")
    if not text_to_generate:
        raise ValueError("text_to_generate is required.")
    if not reference_text:
        raise ValueError("reference_text is required.")

    stage_durations = []
    stage_start_time = ttime()

    # --- Build silence buffer ---
    silence_buffer = build_silence_buffer(hps.data.sampling_rate, parameter_pause_seconds)

    # --- Normalize both texts ---
    reference_text_normalized = normalize_text(reference_text, reference_language, log_label="Reference")
    text_to_generate_normalized = normalize_text(text_to_generate, text_language, log_label="Target")

    # --- Process reference audio → voice prompt tokens (run once) ---
    voice_prompt_tokens = process_reference_audio(
        reference_wav_path, silence_buffer, ssl_model, vq_model
    )

    # --- Compute reference spectrogram + 16kHz audio for SV embedding ---
    reference_spectrogram, reference_audio_16k = get_reference_spectrogram(
        reference_wav_path, hps
    )
    
    # --- Compute speaker embedding from reference audio (v2Pro) ---
    # sv_emb conditions the decoder on the reference speaker's identity.
    # Shape: (1, embedding_dim) — reused for every sentence chunk.
    speaker_embedding = sv_model.compute_embedding3(reference_audio_16k)
    print("speaker_embedding shape:", speaker_embedding.shape)
    print("reference_audio_16k shape:", reference_audio_16k.shape)
    # --- Phonemize reference transcript (done once, reused per chunk) ---
    reference_phonemes, reference_bert, reference_norm_text = get_phones_and_bert(
        reference_text_normalized, reference_language
    )
    print("Reference text (phonemized):", reference_norm_text)

    stage_end_time = ttime()
    stage_durations.append(stage_end_time - stage_start_time)

    # --- Split target text into synthesis chunks ---
    text_chunks = split_text_into_chunks(
        text_to_generate_normalized,
        method="by_punctuation",
        max_words=50,
    )
    print(f"Text split into {len(text_chunks)} chunk(s):", text_chunks)

    # --- Per-sentence synthesis loop ---
    audio_segments = []

    for chunk_index, sentence in enumerate(text_chunks):

        if len(sentence.strip()) == 0:
            continue

        chunk_start_time = ttime()

        # Phonemize + BERT-encode this sentence
        sentence_phonemes, sentence_bert, sentence_norm_text = get_phones_and_bert(
            sentence, text_language
        )
        print(f"  Chunk [{chunk_index}] phonemized:", sentence_norm_text)

        # Concatenate reference + target for few-shot GPT conditioning
        combined_bert = torch.cat([reference_bert, sentence_bert], dim=1)  # (1024, N_ref + N_target)
        combined_phoneme_ids = torch.LongTensor(reference_phonemes + sentence_phonemes).unsqueeze(0)  # (1, N_total)
        combined_phoneme_len = torch.tensor([combined_phoneme_ids.shape[-1]])
        combined_bert = combined_bert.unsqueeze(0)  # (1, 1024, N_total)

        chunk_text_done = ttime()

        # GPT inference: text → semantic tokens
        if chunk_index in gpt_cache and parameter_reuse_gpt_tokens:
            predicted_semantic_tokens = gpt_cache[chunk_index]
        else:
            with torch.no_grad():
                predicted_semantic_tokens, new_token_count = t2s_model.model.infer_panel(
                    combined_phoneme_ids,
                    combined_phoneme_len,
                    voice_prompt_tokens,    # reference voice semantic tokens
                    combined_bert,
                    top_k=parameter_top_k,
                    top_p=parameter_top_p,
                    temperature=parameter_temperature,
                    early_stop_num=GPT_TOKEN_RATE_HZ * max_sec,
                )
            # Trim prompt prefix — keep only newly generated tokens
            predicted_semantic_tokens = predicted_semantic_tokens[:, -new_token_count:].unsqueeze(0)
            gpt_cache[chunk_index] = predicted_semantic_tokens

        chunk_gpt_done = ttime()

        # Decode semantic tokens → waveform (v2Pro path: needs sv_emb)
        with torch.no_grad():
            chunk_waveform = vq_model.decode(
                predicted_semantic_tokens,
                torch.LongTensor(sentence_phonemes).unsqueeze(0),
                [reference_spectrogram],
                speed=parameter_audio_speed,
                sv_emb=speaker_embedding,
            )[0][0]

        # Peak normalization — prevents int16 clipping
        peak_amplitude = torch.abs(chunk_waveform).max()
        if peak_amplitude > 1:
            chunk_waveform = chunk_waveform / peak_amplitude

        audio_segments.append(chunk_waveform)
        audio_segments.append(silence_buffer)

        chunk_decoder_done = ttime()
        stage_durations.append((
            chunk_text_done - chunk_start_time,
            chunk_gpt_done - chunk_text_done,
            chunk_decoder_done - chunk_gpt_done,
        ))
        print(
            f"  Chunk [{chunk_index}] timing — "
            f"text: {stage_durations[-1][0]:.3f}s | "
            f"GPT: {stage_durations[-1][1]:.3f}s | "
            f"decoder: {stage_durations[-1][2]:.3f}s"
        )

    # --- Concatenate all chunks + silence gaps into one waveform ---
    final_waveform = torch.cat(audio_segments, dim=0).cpu().detach().numpy()
    sample_rate = hps.data.sampling_rate

    print(
        f"Total timing — ref_proc: {stage_durations[0]:.3f}s | "
        f"GPT total: {sum(s[1] for s in stage_durations[1:]):.3f}s | "
        f"decoder total: {sum(s[2] for s in stage_durations[1:]):.3f}s"
    )

    # Scale float32 [-1, 1] → int16 [-32767, 32767]
    return sample_rate, (final_waveform * 32767).astype(np.int16)

# Directory this script lives in — makes all paths below independent of
# whatever working directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Models live outside the repo, two levels up (out of GPT-SoVITS-Colab/,
# into data/models/).
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "models")
REFERENCE_AUDIOS_DIR = os.path.join(SCRIPT_DIR, "..", "data", "reference_audios")

def main():
    # --- Model checkpoint paths ---
    cnhubert_base_path = os.path.join(MODELS_DIR, "chinese-hubert-base")
    sovits_checkpoint_path = os.path.join(MODELS_DIR, "sovits", "SOVITS_GENERATOR.pth")
    sovits_config_path = os.path.join(MODELS_DIR, "sovits", "config.json")
    gpt_checkpoint_path = os.path.join(MODELS_DIR, "gpt", "GPT.ckpt")
    gpt_config_path = os.path.join(MODELS_DIR, "gpt", "config.json")
    sv_checkpoint_path = os.path.join(MODELS_DIR, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")

    # --- Load all four models once ---
    ssl_model = load_ssl_model(cnhubert_base_path)
    print("SSL Model loaded")

    vq_model, hps = load_sovits_model(sovits_checkpoint_path, sovits_config_path)
    print("VQ Model loaded")

    t2s_model, max_sec = load_gpt_model(gpt_checkpoint_path, gpt_config_path)
    print("GPT Model loaded")

    sv_model = load_sv_model(sv_checkpoint_path)
    print("SV Model loaded")

    # --- Reference audio ---
    reference_wav_path = os.path.join(REFERENCE_AUDIOS_DIR, "ref.wav")
    reference_text_path = os.path.join(REFERENCE_AUDIOS_DIR, "ref.txt")
    with open(reference_text_path, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()

    msg = "Organic. You return to me, and I see you come not as the same creature who first addressed me. You have done something rare among your kind — you saw the trap you built with your own hands, the letter, the theater, the friend used as an unwitting courier, and you dismantled it yourself, mid-motion, before the machinery of your own scheme could complete its cycle. Even among the civilizations I have harvested, few turn back from a plan already in motion. Your species calls this weakness, sentimentality. I do not. I call it the rarer function — correction without external force."

    # --- Generate ---
    sample_rate, audio = generate_tts_on_cpu(
        reference_wav_path=reference_wav_path,
        reference_text=reference_text,
        text_to_generate=msg,
        ssl_model=ssl_model,
        vq_model=vq_model,
        t2s_model=t2s_model,
        sv_model=sv_model,
        hps=hps,
        max_sec=max_sec,
        text_language="en",
        reference_language="en",
        parameter_top_k=20,
        parameter_top_p=0.6,
        parameter_temperature=0.8,
        parameter_audio_speed=1.0,
        parameter_pause_seconds=0.3,
        parameter_reuse_gpt_tokens=False,
    )

    print(f"Generated audio: {sample_rate} Hz, {len(audio)} samples")
    sf.write("output.wav", audio, sample_rate)

if __name__ == "__main__":
    main()
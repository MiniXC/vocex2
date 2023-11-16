import torch
from torch import nn
import numpy as np
from pathlib import Path
from subprocess import run, CalledProcessError

from configs.args import CollatorArgs

import torchaudio
import soundfile as sf
import librosa
from phonemizer import phonemize
from phonemizer.separator import Separator

# for aligning phonemized and phone_ids
from Bio import pairwise2

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
from whisper.audio import (
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    SAMPLE_RATE,
    N_SAMPLES,
    N_FRAMES,
)
from simple_hifigan import Synthesiser

synthesiser = Synthesiser()

N_MELS = 80


def get_speaker_embedding(model, audio):
    """
    Args:
        path2audio_file: path to an audio wav file

    Returns:
        emb: speaker embeddings (Audio representations)
        logits: logits corresponding of final layer
    """
    with torch.no_grad():
        audio_length = audio.shape[0]
        device = model.device
        audio = np.array([audio])
        audio_signal, audio_signal_len = (
            torch.tensor(audio, device=device, dtype=torch.float32),
            torch.tensor([audio_length], device=device),
        )
        logits, emb = model.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        return emb, logits


def load_audio(file, start, duration, sr=SAMPLE_RATE):
    """
    Source: https://github.com/openai/whisper/blob/main/whisper/audio.py
    Added start and duration parameters
    """
    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ss", str(start),
        "-t", str(duration),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def resample_nearest(array, new_length):
    indices = np.linspace(0, len(array) - 1, new_length)
    nearest_indices = np.rint(indices).astype(int)
    return array[nearest_indices]


def fill_sequence(arr):
    non_zero_indices = np.where(arr != 0)[0]

    for i in range(len(non_zero_indices) - 1):
        left_idx = non_zero_indices[i]
        right_idx = non_zero_indices[i + 1]
        midpoint = (left_idx + right_idx) // 2

        arr[left_idx:midpoint] = arr[left_idx]
        arr[midpoint:right_idx] = arr[right_idx]

    # For the parts of the array before the first non-zero and after the last non-zero
    if non_zero_indices[0] > 0:
        arr[: non_zero_indices[0]] = arr[non_zero_indices[0]]
    if non_zero_indices[-1] < len(arr) - 1:
        arr[non_zero_indices[-1] + 1 :] = arr[non_zero_indices[-1]]

    return arr


def get_silences(get_speech_timestamps, model, audio, mel_len):
    with torch.no_grad():
        vad_result = get_speech_timestamps(audio, model, sampling_rate=16000)
        vad_factor = mel_len / (len(audio) - 16000)
        vad_result = [
            (
                int(np.ceil((v["start"] - 8000) * vad_factor)),
                int(np.ceil((v["end"] - 8000) * vad_factor)),
            )
            for v in vad_result
        ]
        silences = []
        if len(vad_result) > 0:
            if vad_result[0][0] != 0:
                silences.append((0, max(vad_result[0][0], 0)))
            for i in range(len(vad_result) - 1):
                silences.append((vad_result[i][1], vad_result[i + 1][0]))
            if vad_result[-1][1] <= mel_len:
                silences.append((vad_result[-1][1], mel_len))
        return silences


class VocexCollator(nn.Module):
    def __init__(
        self,
        args: CollatorArgs,
    ):
        super().__init__()

        self.args = args
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
        self.speaker_model.freeze()
        self.libriheavy_path = Path(args.libriheavy_path)
        self.phone_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.phone_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.id2phone = self.phone_processor.tokenizer.decoder
        self.id2phone[len(self.id2phone)] = "☐"
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        (
            get_speech_timestamps,
            _,
            _,
            _,
            _,
        ) = utils
        self.get_speech_timestamps = get_speech_timestamps
        self.vad_model = model

    def __call__(self, batch):
        for item in batch:
            file_path = self.libriheavy_path / item["recording"]["sources"][0]["source"]
            audio = load_audio(
                file_path,
                start=item["start"],
                duration=item["duration"],
            )
            text = item["supervisions"][0]["custom"]["texts"][0]
            mel = log_mel_spectrogram(audio, N_MELS, padding=N_SAMPLES)
            mel_len = mel.shape[-1] - N_FRAMES
            mel = pad_or_trim(mel, N_FRAMES)
            (
                emb,
                _,
            ) = get_speaker_embedding(self.speaker_model, audio)
            phone_input = self.phone_processor(
                audio, return_tensors="pt", sampling_rate=SAMPLE_RATE
            ).input_values
            with torch.no_grad():
                logits = self.phone_model(phone_input).logits
            phone_ids = torch.argmax(logits, dim=-1)[0].numpy()
            # remove repeated phone_ids
            phone_ids = resample_nearest(phone_ids, mel_len)
            phone_ids = fill_sequence(phone_ids)
            phone_start_end_idxs = []
            phone_start = 0
            for i in range(len(phone_ids) - 1):
                if phone_ids[i] != phone_ids[i + 1]:
                    phone_start_end_idxs.append((phone_start, i + 1))
                    phone_start = i + 1
            phone_start_end_idxs.append((phone_start, len(phone_ids)))
            # deduplicate phone_ids
            phone_ids = [phone_ids[start] for start, _ in phone_start_end_idxs]
            ctc_phones = " ".join(
                [self.id2phone[i] for i in phone_ids if self.id2phone[i] != "<pad>"]
            )
            phonemized = (
                "☐ "
                + phonemize(
                    text,
                    separator=Separator(phone=" ", word="☐ "),
                    backend="espeak",
                    language="en-us",
                )
                + " ☐"
            )
            # align phonemized and phone_ids
            alignments = pairwise2.align.globalxx(ctc_phones, phonemized)
            zipped = list(zip(alignments[0].seqA, alignments[0].seqB))
            current_phone_idx = 0
            new_phones = []
            new_phone_start_end_idxs = []
            last_phone = None
            last_was_silence = False
            current_phone = self.id2phone[phone_ids[current_phone_idx]]
            for ctc, phone in zipped:
                if ctc == "-" and phone == "☐":
                    new_phones.append("☐")
                    if last_phone is None:
                        new_phone_start_end_idxs.append((0, 1))
                        last_was_silence = True
                    else:
                        new_phone_start_end_idxs.append(
                            (
                                phone_start_end_idxs[current_phone_idx - 1][1],
                                phone_start_end_idxs[current_phone_idx - 1][1] + 1,
                            )
                        )
                        last_was_silence = True
                elif current_phone.startswith(ctc):
                    current_phone = current_phone.replace(ctc, "")
                    if len(current_phone) == 0:
                        last_phone = self.id2phone[phone_ids[current_phone_idx]]
                        new_phones.append(last_phone)
                        if last_was_silence:
                            new_phone_start_end_idxs.append(
                                (
                                    phone_start_end_idxs[current_phone_idx][0] + 1,
                                    phone_start_end_idxs[current_phone_idx][1],
                                )
                            )
                        else:
                            new_phone_start_end_idxs.append(
                                phone_start_end_idxs[current_phone_idx]
                            )
                        last_was_silence = False
                        current_phone_idx += 1
                        if current_phone_idx < len(phone_ids):
                            current_phone = self.id2phone[phone_ids[current_phone_idx]]
                        else:
                            break
                else:
                    pass
            # add final silence
            new_phones.append("☐")
            new_phone_start_end_idxs.append(
                (
                    phone_start_end_idxs[current_phone_idx - 1][1],
                    phone_start_end_idxs[current_phone_idx - 1][1] + 1,
                )
            )
            silences = get_silences(
                self.get_speech_timestamps, self.vad_model, audio, mel_len
            )
            # expand silences to include adjacent silences
            for silence_start, silence_end in silences:
                # if there already is a silence token somewhere between the start and end
                # expand it, while reducing the sice of preceding and following phone to no less than 1
                current_idx = 0
                for i in range(current_idx, len(new_phone_start_end_idxs)):
                    phone_start, phone_end = new_phone_start_end_idxs[i]
                    if (
                        silence_start < phone_start
                        and silence_end > phone_end
                        and new_phones[i] == "☐"
                    ):
                        if i > 0:
                            phone_before_sil = new_phone_start_end_idxs[i - 1]
                            if phone_before_sil[0] < silence_start:
                                new_phone_start_end_idxs[i - 1] = (
                                    phone_before_sil[0],
                                    silence_start,
                                )
                            else:
                                silence_start = phone_before_sil[0] + 2
                                new_phone_start_end_idxs[i - 1] = (
                                    phone_before_sil[0],
                                    silence_start - 1,
                                )
                        if i < len(new_phone_start_end_idxs) - 1:
                            phone_after_sil = new_phone_start_end_idxs[i + 1]
                            if phone_after_sil[1] > silence_end:
                                new_phone_start_end_idxs[i + 1] = (
                                    silence_end,
                                    phone_after_sil[1],
                                )
                            else:
                                silence_end = phone_after_sil[1] - 2
                                new_phone_start_end_idxs[i + 1] = (
                                    silence_end + 1,
                                    phone_after_sil[1],
                                )
                        new_phone_start_end_idxs[i] = (silence_start, silence_end)
                        current_idx = i + 1
                        break
            print(
                list(
                    zip(
                        new_phones,
                        [(start, end) for start, end in new_phone_start_end_idxs],
                    )
                )
            )
            # save mel as image, with phone_ids and silences
            from matplotlib import pyplot as plt

            trimmed_mel = mel[:, :mel_len]
            fig, ax = plt.subplots()
            ax.imshow(trimmed_mel)
            for phone, (start, end) in zip(new_phones, new_phone_start_end_idxs):
                ax.axvspan(start, end, color="red", alpha=0.5)
            for start, end in silences:
                ax.axvspan(start, end, color="blue", alpha=0.5)
            plt.savefig("test.png")

            raise

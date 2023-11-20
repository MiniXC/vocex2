import re
from pathlib import Path
from subprocess import run, CalledProcessError
import logging

import torch
from torch import nn
import numpy as np

from configs.args import CollatorArgs

import torchaudio
import soundfile as sf
import librosa
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.punctuation import Punctuation

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


def get_speech_timestamps(
    audio,
    model,
    threshold=0.5,
    sampling_rate=16000,
    min_speech_duration_ms=250,
    max_speech_duration_s=float("inf"),
    min_silence_duration_ms=100,
    window_size_samples=512,
    speech_pad_ms=30,
    return_seconds=False,
    visualize_probs=False,
    progress_tracking_callback=None,
):
    """
    From: https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                "More than one dimension in audio. Are you trying to process audio with 2 channels?"
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn(
            "Sampling rate is a multiply of 16000, casting to 16000 manually!"
        )
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn(
            "window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!"
        )
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn(
            "Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate"
        )

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, int(window_size_samples - len(chunk)))
            )
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)
        # caculate progress and seng it to callback function
        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = (
        next_start
    ) = 0  # to save potential segment limits in case of maximum segment size reached

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if (
                    next_start < prev_end
                ):  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (
                (window_size_samples * i) - temp_end
            ) > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


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
        audio = torch.cat(
            [torch.zeros(1000), torch.tensor(audio), torch.zeros(1000)], dim=0
        )
        vad_result = get_speech_timestamps(audio, model, sampling_rate=16000)
        vad_factor = mel_len / (len(audio) - 2000)
        vad_result = [
            (
                int(np.ceil((v["start"] - 1000) * vad_factor)),
                int(np.ceil((v["end"] - 1000) * vad_factor)),
            )
            for v in vad_result
        ]
        silences = []
        frame_pad = 2
        if len(vad_result) > 0:
            if vad_result[0][0] != 0:
                silences.append((0, max(vad_result[0][0] + frame_pad, 0)))
            for i in range(len(vad_result) - 1):
                silences.append(
                    (vad_result[i][1] - frame_pad, vad_result[i + 1][0] + frame_pad)
                )
            if vad_result[-1][1] <= mel_len:
                silences.append((vad_result[-1][1] - frame_pad, mel_len + frame_pad))
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
        self.punct_symbols = ';:,.!"?()-'
        for symbol in self.punct_symbols:
            self.id2phone[len(self.id2phone)] = symbol
        self.id2phone[len(self.id2phone)] = "☐"
        self.phone2id = {v: k for k, v in self.id2phone.items()}
        vad_model = torch.jit.load("collators/silero_vad.jit")
        self.get_speech_timestamps = get_speech_timestamps
        self.vad_model = vad_model
        self.phone_len = args.phone_len
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)

    def __call__(self, batch):
        result = {
            "mel": [],
            "mel_len": [],
            "speaker_emb": [],
            "phone_ids": [],
            "phone_spans": [],
            "transcript": [],
            "transcript_phonemized": [],
            "silences": [],
        }
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
                    separator=Separator(phone=" ", word=" ☐ "),
                    backend="espeak",
                    language="en-us",
                    preserve_punctuation=True,
                    punctuation_marks=self.punct_symbols,
                    logger=self.logger,
                )
                + " ☐"
            )
            # if punctuations are not separated by spaces, add spaces
            for symbol in self.punct_symbols:
                phonemized = phonemized.replace(symbol, f" {symbol} ")
            # replace any spaces more than 1 with 1 space
            phonemized = re.sub(r"\s{2,}", " ", phonemized)
            # remove 2 consecutive punctuations, or space and punctuation
            phonemized_temp = []
            phonemized_split = phonemized.split(" ")
            for i in range(len(phonemized_split) - 1):
                if (
                    phonemized_split[i] == "☐"
                    and phonemized_split[i + 1] in self.punct_symbols
                ):
                    continue
                elif (
                    phonemized_split[i] in self.punct_symbols
                    and phonemized_split[i + 1] == "☐"
                ):
                    phonemized_split[i + 1] = phonemized_split[i]
                else:
                    phonemized_temp.append(phonemized_split[i])
            phonemized_temp.append(phonemized_split[-1])
            phonemized = " ".join(phonemized_temp)
            phonemized_ids = []

            for phone in phonemized.split(" "):
                if len(phone) > 0:
                    if phone not in self.phone2id:
                        # find any substring of phone that is in phone2id
                        found_substring = False
                        for i in range(len(phone)):
                            if phone[: len(phone) - i] in self.phone2id:
                                phonemized_ids.append(
                                    self.phone2id[phone[: len(phone) - i]]
                                )
                                found_substring = True
                                break
                        if not found_substring:
                            for i in range(len(phone)):
                                if phone[i:] in self.phone2id:
                                    phonemized_ids.append(self.phone2id[phone[i:]])
                                    found_substring = True
                                    break
                        if not found_substring:
                            print(f"Could not find {phone} in phone2id")
                            phonemized_ids.append(0)
                    else:
                        phonemized_ids.append(self.phone2id[phone])
            # align phonemized and phone_ids
            alignments = pairwise2.align.globalxx(ctc_phones, phonemized, gap_char="+")
            zipped = list(zip(alignments[0].seqA, alignments[0].seqB))
            current_phone_idx = 0
            new_phones = []
            new_phone_start_end_idxs = []
            last_phone = None
            last_was_silence = False
            current_phone = self.id2phone[phone_ids[current_phone_idx]]
            silence_and_punct = "☐" + self.punct_symbols
            for ctc, phone in zipped:
                if ctc == "+" and phone in silence_and_punct:
                    new_phones.append(phone)
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
            if new_phones[-1] not in silence_and_punct:
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
                        silence_start <= phone_start
                        and silence_end >= phone_end
                        and new_phones[i] in silence_and_punct
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
                                    silence_start,
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
                                    silence_end,
                                    phone_after_sil[1],
                                )
                        new_phone_start_end_idxs[i] = (silence_start, silence_end)
                        current_idx = i + 1
                        break

            phone_spans = list(
                zip(
                    new_phones,
                    [(start, end) for start, end in new_phone_start_end_idxs],
                )
            )
            if phone_spans[0][1][0] < 0:
                phone_spans[0] = (
                    phone_spans[0][0],
                    (0, phone_spans[0][1][1]),
                )
            if phone_spans[-1][1][1] > mel_len:
                phone_spans[-1] = (
                    phone_spans[-1][0],
                    (phone_spans[-1][1][0], mel_len),
                )
            # remove duplicate silences
            phone_spans_temp = []
            prev_str = ""
            # ';:,.!"?()-' ordered from least to most important for prosody
            punct_importance_order = '?!.,";:-()'
            for i in range(len(phone_spans) - 1):
                if (
                    phone_spans[i][0] in silence_and_punct
                    and phone_spans[i + 1][0] in silence_and_punct
                ):
                    prev_str += phone_spans[i][0]
                else:
                    if prev_str != "":
                        prev_str += phone_spans[i][0]
                        # get the most important punctuation
                        most_important_punct = None
                        for symbol in punct_importance_order:
                            if symbol in prev_str:
                                most_important_punct = symbol
                                break
                        if most_important_punct is None:
                            most_important_punct = "☐"
                        new_start = phone_spans[i - len(prev_str) + 1][1][0]
                        new_end = phone_spans[i][1][1]
                        phone_spans_temp.append(
                            (
                                most_important_punct,
                                (new_start, new_end),
                            )
                        )
                    else:
                        phone_spans_temp.append(phone_spans[i])
                    prev_str = ""
            phone_spans_temp.append(phone_spans[-1])
            phone_spans = phone_spans_temp

            result["phone_spans"].append(phone_spans)
            result["mel"].append(mel)
            result["mel_len"].append(mel_len)
            result["speaker_emb"].append(emb)
            result["transcript"].append(text)
            # pad phonemized_ids to phone_len
            phonemized_ids = phonemized_ids + [0] * (
                self.phone_len - len(phonemized_ids)
            )
            # if phonemized_ids is longer than phone_len, truncate it
            phonemized_ids = phonemized_ids[: self.phone_len]
            result["transcript_phonemized"].append(phonemized_ids)
            result["silences"].append(silences)
            # convert phone_spans to phone_ids
            phone_ids = []
            i = 0
            for phone, (start, end) in phone_spans:
                if i > 0:
                    if start != phone_spans[i - 1][1][1]:
                        print(f"start {start} != {phone_spans[i - 1][1][1]}", phone)
                if i < len(phone_spans) - 1:
                    if end != phone_spans[i + 1][1][0]:
                        print(f"end {end} != {phone_spans[i + 1][1][0]}", phone)
                i += 1
                try:
                    phone_ids.extend([self.phone2id[phone]] * (end - start))
                except KeyError:
                    print(result)
                    print(f"KeyError: {phone}")
                    raise
            assert len(phone_ids) == mel_len
            # pad phone_ids to N_FRAMES
            phone_ids = phone_ids + [0] * (N_FRAMES - len(phone_ids))
            result["phone_ids"].append(phone_ids)
        result["mel"] = torch.stack(result["mel"])
        result["mel_len"] = torch.tensor(result["mel_len"])
        result["speaker_emb"] = torch.stack(result["speaker_emb"])
        result["phone_ids"] = torch.tensor(result["phone_ids"])
        result["transcript_phonemized"] = torch.tensor(result["transcript_phonemized"])
        return result

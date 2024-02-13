import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import math

class MelSpecDataset(torch.utils.data.Dataset):
    
    def __init__(self, *, 
        
        # Audio Parameters
        files, segment_size, 

        # Mel Spectogram Parameters
        mel_sample_rate, 
        mel_n, 
        mel_fft, 
        mel_hop_size, 
        mel_win_size,

        # Output Mel Spectogram Parameters
        output_mel_n,
        output_mel_fft,
        output_mel_hop_size,
        output_mel_win_size,
        output_sample_rate,
    ):
        self.files = files
        self.segment_size = segment_size
        self.mel_n = mel_n
        self.mel_fft = mel_fft
        self.mel_hop_size = mel_hop_size
        self.mel_win_size = mel_win_size
        self.mel_sample_rate = mel_sample_rate
        self.output_mel_n = output_mel_n
        self.output_mel_fft = output_mel_fft
        self.output_mel_hop_size = output_mel_hop_size
        self.output_mel_win_size = output_mel_win_size
        self.output_sample_rate = output_sample_rate
    
    def __getitem__(self, index):

        # Load File
        filename = self.files[index]

        # Load audio
        output_audio = load_mono_audio(filename, self.output_sample_rate)

        # Pad or trim to target duration
        if output_audio.shape[0] >= self.segment_size:
            audio_start = random.randint(0, output_audio.shape[0] - self.segment_size)
            output_audio = output_audio[audio_start:audio_start+self.segment_size]
        elif output_audio.shape[0] < self.segment_size: # Rare or impossible case - just pad with zeros
            output_audio = torch.nn.functional.pad(output_audio, (0, self.segment_size - output_audio.shape[0]))

        # Output Spectogram
        output_spec = spectogram(output_audio, self.output_mel_fft, self.output_mel_n, self.output_mel_hop_size, self.output_mel_win_size, self.output_sample_rate)

        # Compute Spectogram
        input_audio = resampler(self.output_sample_rate, self.mel_sample_rate, output_audio.device)(output_audio)
        spec = spectogram(input_audio, self.mel_fft, self.mel_n, self.mel_hop_size, self.mel_win_size, self.mel_sample_rate)

        return (output_audio, output_spec, spec)

    def __len__(self):
        return len(self.files)
    

#
# Cached Hann Window
#

hann_window_cache = {}
def hann_window(size, device):
    global hann_window_cache
    key = str(device) + "_" + str(size)
    if key in hann_window_cache:
        return hann_window_cache[key]
    else:
        res = torch.hann_window(size).to(device)
        hann_window_cache[key] = res
        return res

#
# Mel Log Bank
#

melscale_fbank_cache = {}
def melscale_fbanks(n_mels, n_fft, f_min, f_max, sample_rate, device):
    global melscale_fbank_cache
    key = str(n_mels) + "_" + str(n_fft) + "_" + str(f_min) + "_" + str(f_max) + "_" + str(sample_rate) + "_" + str(device)
    if key in melscale_fbank_cache:
        return melscale_fbank_cache[key]
    else:
        res = F.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
        ).transpose(-1, -2).to(device)
        melscale_fbank_cache[key] = res
        return res

#
# Resampler
#

resampler_cache = {}
def resampler(from_sample_rate, to_sample_rate, device=None):
    global resampler_cache
    if device is None:
        device = "cpu"
    key = str(from_sample_rate) + "_" + str(to_sample_rate) + "_" + str(device)
    if key in resampler_cache:
        return resampler_cache[key]
    else:
        res = T.Resample(from_sample_rate, to_sample_rate).to(device)
        resampler_cache[key] = res
        return res

#
# Spectogram caclulcation
#

def spectogram(audio, n_fft, n_mels, n_hop, n_window, sample_rate):

    # Hann Window
    window = hann_window(n_window, audio.device)

    # STFT
    stft = torch.stft(audio, 
        
        # STFT Parameters
        n_fft = n_fft, 
        hop_length = n_hop, 
        win_length = n_window,
        window = window, 
        center = True,
        
        onesided = True, # Default to true to real input, but we enforce it just in case
        return_complex = True
    )

    # Compute magnitudes (|a + ib| = sqrt(a^2 + b^2)) instead of power spectrum (|a + ib|^2 = a^2 + b^2)
    # because magnitude and phase is linear to the input, while power spectrum is quadratic to the input
    # and the magnitude is easier to learn for vocoder
    # magnitudes = stft[..., :-1].abs() ** 2 # Power
    magnitudes = stft[..., :-1].abs() # Amplitude

    # Mel Log Bank
    mel_filters = melscale_fbanks(n_mels, n_fft, 0, sample_rate / 2, sample_rate, audio.device)
    mel_spec = (mel_filters @ magnitudes)

    # Log
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()

    return log_spec


#
# Load Mono Audio
#

def load_mono_audio(src, sample_rate, device=None):

    # Load audio
    audio, sr = torchaudio.load(src)

    # Move to device
    if device is not None:
        audio = audio.to(device)

    # Resample
    if sr != sample_rate:
        audio = resampler(sr, sample_rate, device)(audio)
        sr = sample_rate

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    return audio

#
# VAD
#

vad = None

def init_vad_if_needed():
    global vad
    if vad is None:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        vad = (model, utils)
    else:
        model, utils = vad
    return vad

def trim_silence(audio, sample_rate, padding = 0.25):

    # Load VAD
    model, utils = init_vad_if_needed()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Get speech timestamps
    padding_frames = math.floor(sample_rate * padding)
    speech_timestamps = get_speech_timestamps(audio.unsqueeze(0), model.to(audio.device), sampling_rate=sample_rate)    
    if len(speech_timestamps) > 0:
        voice_start = speech_timestamps[0]['start'] - padding_frames
        voice_end = speech_timestamps[-1]['end'] + padding_frames
        voice_start = max(0, voice_start)
        voice_end = min(len(audio), voice_end)
        audio = audio[voice_start:voice_end]

    return audio
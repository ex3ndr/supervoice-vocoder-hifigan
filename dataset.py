import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random

class MelSpecDataset(torch.utils.data.Dataset):
    
    def __init__(self, *, files, sample_rate, segment_size, mel_n, mel_fft, mel_hop_size, mel_win_size):
        self.files = files
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.mel_n = mel_n
        self.mel_fft = mel_fft
        self.mel_hop_size = mel_hop_size
        self.mel_win_size = mel_win_size
    
    def __getitem__(self, index):

        # Load File
        filename = self.files[index]

        # Load audio
        audio = load_mono_audio(filename, self.sample_rate)

        # Pad or trim to target duration
        if audio.shape[0] >= self.segment_size:
            audio_start = random.randint(0, audio.shape[0] - self.segment_size)
            audio = audio[audio_start:audio_start+self.segment_size]
        elif audio.shape[0] < self.segment_size: # Rare or impossible case - just pad with zeros
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[0]))

        # Compute Spectogram
        spec = spectogram(audio, self.mel_fft, self.mel_n, self.mel_hop_size, self.mel_win_size, self.sample_rate)

        return (audio, spec)

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
        n_fft, 
        hop_length=n_hop, 
        win_length=n_window,
        window=window, 
        return_complex=False
    )

    # Compute magnitudes using squared value
    magnitudes = torch.sum((stft ** 2), dim=-1)[..., :-1]

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


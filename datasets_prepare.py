# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import csv
import os
import multiprocessing
import glob
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from dataset import load_mono_audio, spectogram, init_vad_if_needed, trim_silence

#
# Parameters
#

PARAM_WORKERS = torch.cuda.device_count() * 4

#
# Execution
#

def speaker_directory(speaker):
    return str(speaker).zfill(8)

def execute_parallel(args):
    process_id = multiprocessing.current_process()._identity[0]
    files, destination, use_vad, index = args
    file = files[index]
    device = "cuda:" + str(process_id % torch.cuda.device_count())

    # Format filename from index (e.g. 000001)
    target_name = str(index).zfill(8)

    # Load audio
    try:
        waveform = load_mono_audio(file, 16000, device=device)
    except:
        print(f'File {file} ignored')
        return # Ignore error

    # Trim silence
    if use_vad:
        waveform = trim_silence(waveform, 16000)

    # Split to max 10 seconds
    if len(waveform) > 16000 * 10:
        count = len(waveform) // (16000 * 10) + 1
        interval_length = len(waveform) // count
        for i in range(count):
            start = i * interval_length
            end = (i + 1) * interval_length
            if end > len(waveform):
                end = len(waveform)
            target_name_split = target_name + "_" + str(i).zfill(3)
            target_dir = os.path.join(destination, str(index // 1000).zfill(8))
            torchaudio.save(os.path.join(target_dir, target_name_split + ".wav"), waveform[start:end].unsqueeze(0).cpu(), 16000)
    else:
        target_dir = os.path.join(destination, str(index // 1000).zfill(8))
        torchaudio.save(os.path.join(target_dir, target_name + ".wav"), waveform.unsqueeze(0).cpu(), 16000)
    

def load_cv_files():
    # Load CSV
    with open("external_datasets/common-voice-16.0-en/en/train.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) 
        data = list(reader)

    # Extract files
    files = ["external_datasets/common-voice-16.0-en/en/clips/" + d[1] for d in data]

    return files

def execute_run():
    init_vad_if_needed()

    # Indexing files
    print("Build file index...")

    # Collections
    collections = {}
    collections['musan'] = { 'files': glob.glob("external_datasets/musan/music/*/*.wav") + glob.glob("external_datasets/musan/noise/*/*.wav"), 'vad': False }
    collections['libritts-r-clean-360'] = { 'files': glob.glob("external_datasets/libritts-r-clean-360/*/*/*.wav"), 'vad': False }
    collections['libritts-r-clean-100'] = { 'files': glob.glob("external_datasets/libritts-r-clean-100/*/*/*.wav"), 'vad': False }
    collections['libritts-r-other-500'] = { 'files': glob.glob("external_datasets/libritts-r-other-500/*/*/*.wav"), 'vad': False }
    collections['vctk-corpus-0.92'] = { 'files': glob.glob("external_datasets/vctk-corpus-0.92/**/*.flac"), 'vad': True }
    collections['common-voice'] = { 'files': load_cv_files(), 'vad': True }

    # Process collections
    for collection in collections:
        print(f"Processing collection {collection} with {len(collections[collection]['files'])} files")
        name = collection
        files = collections[collection]['files']
        vad = collections[collection]['vad']
        prepared_dir = "datasets/prepared-" + name + "/"

        # Check if exists
        if Path(prepared_dir).exists():
            print(f"Collection {name} already prepared")
            continue

        # Creating directories
        for i in range(len(files) // 1000 + 1):
            target_name = str(i).zfill(8)
            Path(prepared_dir + target_name).mkdir(parents=True, exist_ok=True)

        # Indexes loop
        print("Preparing files...")
        with multiprocessing.Manager() as manager:
            files = manager.list(files)
            args_list = [(files, prepared_dir, vad, i) for i in range(len(files))]
            with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                for _ in tqdm(pool.imap_unordered(execute_parallel, args_list, chunksize=32), total=len(files)):
                    pass
    
    # End
    print("Done")

if __name__ == "__main__":
    execute_run()
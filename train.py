from accelerate import Accelerator
from glob import glob

def main():
    accelerator = Accelerator(log_with="wandb")
    train_files = glob("external_datasets/libritts-r-clean-100/*/*.wav") + glob("external_datasets/libritts-r-clean-360/*/*.wav") + glob("external_datasets/libritts-r-other-500/*/*.wav")

if __name__ == "__main__":
    main()
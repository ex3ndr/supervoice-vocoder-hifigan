# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Core
import itertools
from glob import glob
import shutil
from pathlib import Path
import random

# ML
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

# Local
from dataset import MelSpecDataset, spectogram
from model import HiFiGAN, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, generator_loss

# Model paramerters
vocoder_resblock = "1"
vocoder_upsample_rates = [8,5,3,2]
vocoder_upsample_kernel_sizes = [16,9,5,4] # Looks like it should be ~2x the upsample rate
vocoder_upsample_initial_channel = 512
vocoder_resblock_kernel_sizes = [3,7,11]
vocoder_resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]

vocoder_output_mel_n = 80
vocoder_output_mel_fft = 1024
vocoder_output_mel_hop_size = 240
vocoder_output_mel_win_size = 960
vocoder_output_sample_rate = 24000

vocoder_mel_n = 80
vocoder_mel_fft = 1024
vocoder_mel_hop_size = 160
vocoder_mel_win_size = 640
vocoder_mel_sample_rate = 16000

# Train parameters
train_experiment = "upscale"
train_project = "hifigan"
train_auto_resume = True
train_segment_size = 12000
train_learning_rate = 2e-4
train_adam_b1 = 0.8
train_adam_b2 = 0.99
train_batch_size = 32 # Per GPU
train_steps = 1000000
train_loader_workers = 4
train_save_every = 1000
train_log_every = 1
train_evaluate_every = 200
train_evaluate_batches = 10

# Train
def main():

    # Prepare accelerator
    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    # Prepare dataset
    accelerator.print("Loading dataset...")

    train_files = []
    train_files += glob("datasets/prepared-libritts-r-clean-100/*/*/*.wav")
    train_files += glob("datasets/prepared-libritts-r-clean-360/*/*/*.wav")
    train_files += glob("datasets/prepared-libritts-r-other-500/*/*.wav")
    train_files += glob("datasets/prepared-musan/*/*.wav")
    train_files += glob("datasets/prepared-vctk-corpus-0.92/*/*.wav") 
    train_files += glob("datasets/prepared-common-voice/*/*.wav")
    train_files.sort()
    random.shuffle(train_files)
    
    test_files = []
    test_files += glob("external_datasets/libritts-r/test-clean/*/*/*.wav") 
    test_files += glob("external_datasets/libritts-r/test-other/*/*/*.wav")
    train_files.sort()
    random.shuffle(test_files)
    test_files = test_files[:train_batch_size * train_evaluate_batches]

    dataset_args = { 
        "segment_size": train_segment_size, 

        # Input
        "mel_sample_rate": vocoder_mel_sample_rate, 
        "mel_n": vocoder_mel_n, 
        "mel_fft": vocoder_mel_fft, 
        "mel_hop_size": vocoder_mel_hop_size, 
        "mel_win_size": vocoder_mel_win_size,

        # Output
        "output_mel_n": vocoder_output_mel_n,
        "output_mel_fft": vocoder_output_mel_fft,
        "output_mel_hop_size": vocoder_output_mel_hop_size,
        "output_mel_win_size": vocoder_output_mel_win_size,
        "output_sample_rate": vocoder_output_sample_rate,
    }
    train_dataset = MelSpecDataset(files = train_files, **dataset_args)
    test_dataset = MelSpecDataset(files = test_files, **dataset_args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_loader_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, num_workers=train_loader_workers, pin_memory=True)

    # Prepare model
    accelerator.print("Loading model...")
    steps = 0
    generator = HiFiGAN(
        mels_n=vocoder_mel_n,
        resblock=vocoder_resblock,
        upsample_rates=vocoder_upsample_rates,
        upsample_kernel_sizes=vocoder_upsample_kernel_sizes,
        upsample_initial_channel=vocoder_upsample_initial_channel,
        resblock_kernel_sizes=vocoder_resblock_kernel_sizes,
        resblock_dilation_sizes=vocoder_resblock_dilation_sizes,
    )
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    optim_g = torch.optim.AdamW(generator.parameters(), train_learning_rate, betas=[train_adam_b1, train_adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), train_learning_rate, betas=[train_adam_b1, train_adam_b2])
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=train_steps)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=train_steps)

    # Accelerate
    generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d, train_loader, test_loader = accelerator.prepare(generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d, train_loader, test_loader)
    train_cycle = cycle(train_loader)
    hps = {
        "segment_size": train_segment_size, 
        "learning_rate": train_learning_rate, 
        "adam_b1": train_adam_b1, 
        "adam_b2": train_adam_b2, 
        "batch_size": train_batch_size, 
        "steps": train_steps, 
    }
    accelerator.init_trackers(train_project, config=hps)
    mpd = mpd.to(device)
    msd = msd.to(device)
    generator = generator.to(device)

    # Save and Load
    def save():

        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{steps}.pt")
        torch.save({

            # Model
            'generator': accelerator.get_state_dict(generator), 
            "mpd": accelerator.get_state_dict(mpd),
            "msd": accelerator.get_state_dict(msd),

            # Optimizer
            'steps': steps,
            'optimizer_g': optim_g.state_dict(), 
            'optimizer_d': optim_d.state_dict(), 
            'scheduler_g': scheduler_g.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    def load():
        checkpoint = torch.load(str(output_dir / f"{train_experiment}.pt"), map_location="cpu")

        # Model
        accelerator.unwrap_model(generator).load_state_dict(checkpoint['generator'])
        accelerator.unwrap_model(mpd).load_state_dict(checkpoint['mpd'])
        accelerator.unwrap_model(msd).load_state_dict(checkpoint['msd'])

        # Optimizer
        optim_g.load_state_dict(checkpoint['optimizer_g'])
        optim_d.load_state_dict(checkpoint['optimizer_d'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        steps = checkpoint['steps']

        return steps
    
    # Load
    if train_auto_resume:
        if (output_dir / f"{train_experiment}.pt").exists():
            accelerator.print("Resuming training...")
            steps = load()

    # Train step
    def train_step():
        generator.train()
        mpd.train()
        msd.train()

        # Load batch
        output_audio, output_spec, input_spec = next(train_cycle)
        output_audio = output_audio.unsqueeze(1) # Adding a channel dimension

        # Generate
        y_g_hat = generator(input_spec)

        #
        # Discriminator
        #

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(output_audio, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(output_audio, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        # Train Discriminator
        loss_disc_all = loss_disc_s + loss_disc_f
        optim_d.zero_grad()
        accelerator.backward(loss_disc_all)
        grad_norm_mpd = accelerator.clip_grad_norm_(mpd.parameters(), 1000.)
        grad_norm_msd = accelerator.clip_grad_norm_(msd.parameters(), 1000.)
        optim_d.step()

        #
        # Generator
        #

        # L1 Mel-Spectrogram Loss
        y_g_hat_mel = spectogram(y_g_hat.squeeze(1), vocoder_output_mel_fft, vocoder_output_mel_n, vocoder_output_mel_hop_size, vocoder_output_mel_win_size, vocoder_output_sample_rate)
        loss_mel = F.l1_loss(output_spec, y_g_hat_mel) * 45

        # Discriminator-based losses
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(output_audio, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(output_audio, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        optim_g.zero_grad()
        accelerator.backward(loss_gen_all)
        grad_norm_g = accelerator.clip_grad_norm_(generator.parameters(), 1000.)
        optim_g.step()

        #
        # Update learning rate for next batch
        #

        scheduler_d.step()
        scheduler_g.step()

        return loss_mel, loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_disc_s, loss_disc_f, grad_norm_mpd, grad_norm_msd, grad_norm_g

    # Train Loop
    accelerator.print("Training started at step", steps)
    while steps < train_steps:

        # Do iteration
        loss_mel, loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_disc_s, loss_disc_f, grad_norm_mpd, grad_norm_msd, grad_norm_g = train_step()

        # Update step
        steps = steps + 1

        # Wait for everyone
        accelerator.wait_for_everyone()

        # Evaluate
        if (steps % train_evaluate_every == 0):
            if accelerator.is_main_process:
                accelerator.print("Evaluating...")
            with torch.inference_mode():      
                generator.eval()
                losses = []
                for test_batch in test_loader:
                    output_audio, output_spec, input_spec = test_batch
                    output_audio = output_audio.unsqueeze(1)
                    y_g_hat = generator(input_spec)
                    y_g_hat_mel = spectogram(y_g_hat.squeeze(1), vocoder_output_mel_fft, vocoder_output_mel_n, vocoder_output_mel_hop_size, vocoder_output_mel_win_size, vocoder_output_sample_rate)
                    loss_mel = F.l1_loss(output_spec, y_g_hat_mel) * 45
                    gathered = accelerator.gather(loss_mel).cpu()
                    if len(gathered.shape) == 0:
                        gathered = gathered.unsqueeze(0)
                    losses += gathered.tolist()
                if accelerator.is_main_process:
                    loss = torch.tensor(losses).mean()
                    accelerator.log({"loss_mel_test": loss}, step=steps)
                    accelerator.print(f"Evaluation Loss: {loss}")
                

        # Log
        if accelerator.is_main_process and (steps % train_log_every == 0):
            accelerator.print(f"Step {steps}: Mel Loss: {loss_mel}, Gen S Loss: {loss_gen_s}, Gen F Loss: {loss_gen_f}, FM S Loss: {loss_fm_s}, FM F Loss: {loss_fm_f}, Disc S Loss: {loss_disc_s}, Disc F Loss: {loss_disc_f}")
            accelerator.log({
                "loss_mel": loss_mel, 
                "loss_gen_s": loss_gen_s, 
                "loss_gen_f": loss_gen_f, 
                "loss_fm_s": loss_fm_s, 
                "loss_fm_f": loss_fm_f, 
                "loss_disc_s": loss_disc_s, 
                "loss_disc_f": loss_disc_f,
                'grad_norm_mpd': grad_norm_mpd,
                'grad_norm_msd': grad_norm_msd,
                'grad_norm_g': grad_norm_g,
            }, step=steps)

        # Save 
        if accelerator.is_main_process and (steps % train_save_every == 0):
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()
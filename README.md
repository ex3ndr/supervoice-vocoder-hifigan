# 🎤 Supervoice Vocoder

This is a training scripts of vocoder of Supervoice. Supervoice uses HiFiGAN as a vocoder model, we need to re-train it to make it compatible with more popular Log-Mel spectograms parameters. We have made a small modification - replacing leaky relu with gelu activation function.

## Requirements

This model requires about 1M updates of the network to perform well, this is about 8xGPU for about a week, which costs about 300$ on medium tier cloud provider like paperspace. Dataset is not required to be large and too diverse - vocoders are very good at generalizing. LibriTTS (and "R" variant) is sufficient for the training. We are providing scripts to automatically download required datasets. This script would work only on Linux and won't fly anywhere else.

## Dependencies

To run this model you need to have the following dependencies installed:
```
pip install torch torchaudio accelerate
```

To download datasets you need a [datasets](https://github.com/ex3ndr/datasets) tool installed in your system, then you can download datasets by running:

```bash
datasets sync
```

## Training

This script automatically selects all GPU for training and you can run it by running:

```bash
python train.py
```

# License

Both model and weights are licensed under MIT license
# AI-Generated Image Detection System

Graduation project (Bachelor of Cybersecurity). It compares detectors of AI-generated images on one fixed test set of real photographs and AI-generated images, and fine-tunes a CNN on varied, full-resolution data to address the failure of off-the-shelf detectors on photographs unlike their training data.

## Setup

```bash
pip install -r requirements.txt
```

## Train

`train.py` fine-tunes ConvNeXt-Tiny (`facebook/convnext-tiny-224`) and pushes the model to the Hugging Face Hub.

Training data is a 24,000-image subset of [Community Forensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) (Park and Owens, CVPR 2025), half real and half generated:

| Class | Source | Images |
|---|---|---|
| Generated | 228 community Stable Diffusion models (latent diffusion) | about 9,000 |
| Generated | GLIDE, DeepFloyd IF (pixel diffusion), BigGAN | 3,000 |
| Real | COCO | 4,800 |
| Real | VISION (full-resolution camera photographs) | 1,200 |
| Real | Landscapes HQ | 3,000 |
| Real | FFHQ portraits | 3,000 |

Images flagged NSFW by the dataset are removed. A random 10% is held out for validation.

On a free Google Colab GPU: open `train.ipynb` in Colab, set the runtime to T4 GPU and run all cells.

On your own GPU:

```bash
python train.py --hub-id YOUR_HF_USERNAME/ai-image-detector
```

To check the pipeline in two minutes on a CPU, with nothing uploaded:

```bash
python train.py --dry-run
```

## Evaluate

```bash
python evaluate.py MODEL_ID --real "path/Real Images" --ai "path/AI Images" --name short_name
python evaluate.py --chart
```

Each run writes the metrics, per-image predictions and a confusion matrix to `results/`. "Real" is the positive class.

## Results

Not measured in this repository yet. The fine-tuned model has not been trained, and the baseline detectors (`Manish18/deepfake-detection-model`, a ViT, and `prithivMLmods/Deep-Fake-Detector-Model`, a SigLIP2 fine-tune) have not been re-run here.

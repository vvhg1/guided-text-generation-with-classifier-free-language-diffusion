# Guided Text Generation with Classifier free Language Diffusion

Author: Victor v. Hobe-Gelting

## This repository builds on and is an adaptation of code from:

Diffusion-LM Improves Controllable Text Generation

>Repository: https://github.com/XiangLi1999/Diffusion-LM

>Paper: https://arxiv.org/pdf/2205.14217.pdf

<br/><br/>
Denoising Diffusion Probabilistic Models
>Repository: https://github.com/hojonathanho/diffusion

>Paper: https://arxiv.org/abs/2006.11239

<br/><br/>
Improved Denoising Diffusion Probabilistic Models
>Repository: https://github.com/openai/improved-diffusion

>Paper: https://arxiv.org/abs/2102.09672

<br/><br/>
Diffusion Models Beat GANS on Image Synthesis
>Repository: https://github.com/openai/guided-diffusion

>Paper: https://arxiv.org/abs/2105.05233

<br/><br/>
GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
>Repository: https://github.com/openai/glide-text2im

>Paper: https://arxiv.org/abs/2112.10741

<br/><br/>
GLID-3
>Repository: https://github.com/Jack000/glid-3

<br/><br/>

## Conda Setup:

```python
conda install -c conda-forge mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0
pip install huggingface_hub==0.4.0
pip install wandb
```

---

## Train Diffusion-LM:

`cd improved-diffusion;`

`python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000 --save_interval 50000 --seed 101 --noise_schedule sqrt --in_channel 128 --modality roc-free --submit no --padding_mode pad --app "--predict_xstart True --training_mode e2e --vocab_size 11043 --roc_train datasets/ROCstory " --notes xstart_e2e --bsz 64`


---


## Controllable Text Generation

`python scripts/infill_free.py --model_path 'diffusion_models/diff_roc-free_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_xstart_e2e/model{model epochs}.pt' --eval_task_ 'free_emotion' --use_ddim True --notes "tree_adagrad" --eta 1. --verbose pipe`

---

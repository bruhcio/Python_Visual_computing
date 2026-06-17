<div align="center">

# Python Visual Computing

A conditional variational autoencoder experiment for MNIST-style image generation.

![Status](https://img.shields.io/badge/status-coursework-0A1317?style=for-the-badge)
![Stack](https://img.shields.io/badge/stack-PyTorch%20Lightning-0064E0?style=for-the-badge)
![Topic](https://img.shields.io/badge/topic-CVAE-444950?style=for-the-badge)

</div>

---

## Overview

This repository implements a simple Conditional Variational Autoencoder (CVAE). The model encodes an image with a class condition, samples a latent vector with reparameterization, and decodes it back into a 28x28 grayscale image.

## What It Shows

- Conditional input using one-hot class labels
- Encoder, latent mean/log-variance, and decoder structure
- Reparameterization trick
- Reconstruction loss and KL divergence
- PyTorch Lightning training module
- Gradio-based test interface

## Structure

| File | Role |
| --- | --- |
| `model.py` | CVAE model and training step |
| `dataset.py` | Dataset handling |
| `train.py` | Training entry point |
| `test.py` | Test/inference entry point |
| `gradio_test.py` | Lightweight UI for generated samples |

## Run

```bash
python train.py
python gradio_test.py
```

## Note

This is a coursework-scale model experiment, not a production image-generation service.

---

## 한국어 버전

# Python Visual Computing

MNIST-style image generation을 위한 Conditional Variational Autoencoder 실험 저장소입니다.

## 개요

이 저장소는 간단한 Conditional Variational Autoencoder(CVAE)를 구현합니다. 모델은 image와 class condition을 함께 encoding하고, reparameterization으로 latent vector를 sampling한 뒤, 다시 28x28 grayscale image로 decoding합니다.

## 보여주는 내용

- one-hot class label을 활용한 conditional input
- encoder, latent mean/log-variance, decoder 구조
- reparameterization trick
- reconstruction loss와 KL divergence
- PyTorch Lightning training module
- Gradio 기반 sample 확인 interface

## 구조

| 파일 | 역할 |
| --- | --- |
| `model.py` | CVAE model 및 training step |
| `dataset.py` | dataset handling |
| `train.py` | training entry point |
| `test.py` | test/inference entry point |
| `gradio_test.py` | generated sample 확인용 lightweight UI |

## 실행

```bash
python train.py
python gradio_test.py
```

## 참고

production image-generation service가 아니라 coursework 규모의 model experiment입니다.

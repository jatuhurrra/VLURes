# VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages

![alt text](https://img.shields.io/badge/paper-arXiv:24XX.XXXXX-b31b1b.svg)

<!-- TODO: Update with arXiv link -->

![alt text](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)

<!-- TODO: Update with Hugging Face link -->

![alt text](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)

Official repository for the JMLR paper "VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages" by Jesse Atuhurra, Iqra Ali, Tomoya Iwakura, Hidetaka Kamigaito, and Tatsuya Hiraoka.

# ⚠️ WARNING ⚠️

> [!WARNING]
> ## This repo is under development at this moment

# Abstract
The evaluation of Vision-Language Models (VLMs) is predominantly limited to English-centric benchmarks with short textual contexts, hindering the assessment of their fine-grained reasoning capabilities in diverse linguistic settings. To address this gap, we introduce VLURes, a new multilingual benchmark designed to evaluate the visual and linguistic understanding of VLMs in long-text scenarios. VLURes comprises 1,000 culturally diverse image-text pairs across four languages—English, Japanese, and the low-resource languages Swahili and Urdu—and introduces eight fine-grained vision-language tasks, including a novel task for identifying unrelatedness between modalities. We evaluate ten prominent VLMs on VLURes, analyzing both their direct responses and generated rationales through both automated scoring and human evaluation by native speakers. Our results reveal significant performance disparities across languages, with even the top-performing model, GPT-4o, lagging human performance by 6.7% on average. This gap is substantially larger for open-source models, highlighting critical areas for improvement in multilingual visual reasoning. VLURes provides a crucial new resource for driving the development of more robust and equitable VLMs.

# Key Contributions
- A Novel Multilingual Benchmark (VLURes): We introduce a new benchmark with 1,000 culturally diverse image-text pairs in English, Japanese, Swahili, and Urdu.
- Long-Text Grounding: Unlike prior benchmarks that use short captions, each image in VLURes is embedded in rich, article-length prose, enabling the evaluation of discourse-level understanding.
- New Vision-Language Resources: We provide the first large-scale, multi-task vision-language datasets for the low-resource languages Swahili and Urdu.
- Fine-Grained Task Suite: VLURes includes eight challenging tasks, including a novel unrelatedness task designed to test a model's ability to discard irrelevant information.
- Comprehensive VLM Evaluation: We benchmark ten leading proprietary and open-source VLMs, providing a detailed analysis of their performance, cross-lingual robustness, and the impact of rationales.
- Public Release: We publicly release our complete dataset, code, and evaluation results to facilitate future research.

# The VLURes Benchmark Tasks

The benchmark is structured around eight distinct vision-language tasks designed to probe different facets of multimodal understanding.
Task ID	Task Name	Description
OR	Object Recognition	Identify and categorize all objects present in the image.
SU	Scene Understanding	Describe the overall scene, setting, and activities taking place.
RU	Relationship Understanding	Explain the spatial, functional, or social relationships between entities.
SS	Semantic Segmentation	Divide the image into labeled semantic regions (e.g., sky, buildings).
IC	Image Captioning	Generate a detailed, natural-language description of the image.
ITM	Image-Text Matching	Find specific parts of the text that directly reference the image content.
U	Unrelatedness (Novel Task)	Identify parts of the text that are not represented in the image.
VQA	Visual Question Answering	Answer a natural language question about the image and its textual context.

# Repository Structure
```
.
├── data/
│   └── (This directory will be populated by the download script)
├── paper/
│   └── VLURes.pdf
├── scripts/
│   ├── download_data.py        # Script to download image-text pairs
│   ├── run_inference.py        # Script to run inference with VLMs
│   └── run_evaluation.py       # Script to run the LLM-as-a-judge evaluation
├── results/
│   ├── inference_outputs/      # Pre-computed outputs from the models
│   └── evaluation_scores/      # Pre-computed scores from the LLM-judge
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

# Setup and Installation
# Prerequisites
- Python 3.9+
- CUDA-enabled GPU (for running local models)
- For reproducing the paper PDF: A LaTeX distribution with XeLaTeX.

# Installation Steps
1. Clone the repository:
```
git clone https://github.com/jatuhurra/VLURes.git
cd VLURes
```
2. Create a virtual environment (recommended):
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the required Python packages:
```
pip install -r requirements.txt
```

# Data Download
The VLURes dataset (annotations, text, and image URLs) is hosted on the Hugging Face Hub.

**Important Note on Copyright:** The images in VLURes were collected from public web sources and are subject to their original copyrights. We do not own the copyright to these images and do not distribute them directly. We provide the original URLs and a download script for research purposes. Users are responsible for their own compliance with the images' original license terms.

1. Download the dataset metadata from Hugging Face:
The metadata will be downloaded automatically by the script, but you can also access it here:
https://huggingface.co/datasets/your-username/vlures <!-- TODO: Update with Hugging Face link -->

3. Run the download script to fetch the images:
This script will read the dataset files and download the images from their original URLs into the data/ directory.
```
python scripts/download_data.py

```

# Running the Experiments
To reproduce the results from our paper, follow the steps below.
### 1. Set Up API Keys
For experiments involving proprietary models (GPT-4o, Gemini), you must set your API keys as environment variables:
```
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-api-key"
```
### 2. Run Inference
The run_inference.py script generates responses from the VLMs for all tasks and languages.
```
# Run GPT-4o on the English benchmark for the zero-shot setting without rationales
python scripts/run_inference.py --model gpt-4o --language en --setting zero_shot_no_rat

# Run a local open-source model (e.g., LlaVa)
python scripts/run_inference.py --model llava-mistral-7b --language jp --setting one_shot_with_rat
```
### 3. Run LLM-as-a-Judge Evaluation
The run_evaluation.py script uses Gemini 1.5 Pro to score the generated outputs. This step can be costly (see our paper's Appendix for a full cost analysis).
Example Usage:
```
# Evaluate the outputs generated for GPT-4o on the English benchmark
python scripts/run_evaluation.py --model gpt-4o --language en
```
You can also find our pre-computed model outputs and evaluation scores in the results/ directory to bypass these steps.
# License
The code, annotations, and other original materials in this repository are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

As stated above, the images are not owned by us and are not covered by this license. They are subject to their own original copyright terms.

# Citation
If you use VLURes in your research, please cite our paper:
```
Under review at the Journal of Machine Learning Research!
```
# Contact
For questions about the paper or the repository, please contact the corresponding authors:

- Jesse Atuhurra (atuhurra.jesse.ag2@naist.ac.jp)
- Tatsuya Hiraoka (tatsuya.hiraoka@mbzuai.ac.ae)

# VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages

[![Paper](https://img.shields.io/badge/paper-arXiv:24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX) <!-- TODO: Update with your arXiv link -->
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/atamiles/VLURes)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

> [!WARNING]
> ## This repository is currently under development.
> The code and data are provided as-is to accompany our JMLR submission.

Official repository for the JMLR paper "VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages" by Jesse Atuhurra, Iqra Ali, Tomoya Iwakura, Hidetaka Kamigaito, and Tatsuya Hiraoka.

---

## Abstract
The evaluation of Vision-Language Models (VLMs) is predominantly limited to English-centric benchmarks with short textual contexts, hindering the assessment of their fine-grained reasoning capabilities in diverse linguistic settings. To address this gap, we introduce VLURes, a new multilingual benchmark designed to evaluate the visual and linguistic understanding of VLMs in long-text scenarios. VLURes comprises 1,000 culturally diverse image-text pairs across four languages—English, Japanese, and the low-resource languages Swahili and Urdu—and introduces eight fine-grained vision-language tasks, including a novel task for identifying unrelatedness between modalities. We evaluate ten prominent VLMs on VLURes, analyzing both their direct responses and generated rationales through both automated scoring and human evaluation by native speakers. Our results reveal significant performance disparities across languages, with even the top-performing model, GPT-4o, lagging human performance by 6.7% on average. This gap is substantially larger for open-source models, highlighting critical areas for improvement in multilingual visual reasoning. VLURes provides a crucial new resource for driving the development of more robust and equitable VLMs.

## Key Contributions
- **A Novel Multilingual Benchmark (VLURes):** We introduce a new benchmark with 1,000 culturally diverse image-text pairs in English, Japanese, Swahili, and Urdu.
- **Long-Text Grounding:** Unlike prior benchmarks that use short captions, each image in VLURes is embedded in rich, article-length prose, enabling the evaluation of discourse-level understanding.
- **New Vision-Language Resources:** We provide the first large-scale, multi-task vision-language datasets for the low-resource languages Swahili and Urdu.
- **Fine-Grained Task Suite:** VLURes includes eight challenging tasks, including a novel **unrelatedness** task designed to test a model's ability to discard irrelevant information.
- **Comprehensive VLM Evaluation:** We benchmark ten leading proprietary and open-source VLMs, providing a detailed analysis of their performance, cross-lingual robustness, and the impact of rationales.
- **Public Release:** We publicly release our complete dataset, code, and evaluation results to facilitate future research.

## The VLURes Benchmark Tasks
The benchmark is structured around eight distinct vision-language tasks designed to probe different facets of multimodal understanding.

| Task ID | Task Name                    | Description                                                                 |
| :------ | :--------------------------- | :-------------------------------------------------------------------------- |
| **OR**  | Object Recognition           | Identify and categorize all objects present in the image.                   |
| **SU**  | Scene Understanding          | Describe the overall scene, setting, and activities taking place.           |
| **RU**  | Relationship Understanding   | Explain the spatial, functional, or social relationships between entities.  |
| **SS**  | Semantic Segmentation        | Divide the image into labeled semantic regions (e.g., sky, buildings).      |
| **IC**  | Image Captioning             | Generate a detailed, natural-language description of the image.             |
| **ITM** | Image-Text Matching          | Find specific parts of the text that directly reference the image content.  |
| **U**   | Unrelatedness (Novel Task)   | Identify parts of the text that are **not** represented in the image.       |
| **VQA** | Visual Question Answering    | Answer a natural language question about the image and its textual context. |

## Repository Structure
```
.
├── data/ # (Populated by the download script)
├── paper/
│ └── VLURes.pdf
├── results/
│ ├── inference_outputs/ # Pre-computed model outputs
│ └── evaluation_scores/ # Pre-computed LLM-judge scores
├── scripts/
│ ├── download_data.py
│ ├── run_zeroshot_no_rationales.py
│ ├── run_zeroshot_with_rationales.py
│ ├── run_oneshot_no_rationales.py
│ ├── run_oneshot_with_rationales.py
│ └── run_evaluation.py
├── requirements.txt
├── run_benchmark.sh # Main script to execute experiments
└── README.md
```


---

## How to Reproduce Our Results

### 1. Setup and Installation

#### Prerequisites
- Python 3.9+
- A CUDA-enabled GPU (for running local models)
- For reproducing the paper PDF: A LaTeX distribution with **XeLaTeX**.

#### Installation Steps
1.  **Clone the repository:**
    ```
    git clone https://github.com/jatuhurra/VLURes.git
    cd VLURes
    ```
2.  **Create a virtual environment (recommended):**
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Data Download

The VLURes dataset (annotations, text, and image URLs) is hosted on the Hugging Face Hub.

> **Important Note on Copyright:** The images in VLURes were collected from public web sources and are subject to their original copyrights. We do not own the copyright to these images and do not distribute them directly. We provide the original URLs and a download script for research purposes. Users are responsible for their own compliance with the images' original license terms.

**Run the download script to fetch the images:**
This script will download the dataset from Hugging Face and retrieve all images from their original URLs, saving them to the `data/` directory.
```
python scripts/download_data.py
```

### 3. Running Experiments
Set Up API Keys
For experiments involving proprietary models (e.g., GPT-4o, Gemini), you must set your API keys as environment variables.
```
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-api-key"
```
Running Inference with run_benchmark.sh

We provide a convenient shell script, run_benchmark.sh, to execute any experiment.

**1. Make the script executable:**
```
chmod +x run_benchmark.sh
```

**2. Run a specific experiment:**
The script takes three arguments: <language>, <task_number>, and <setting>.
```
# Example: Run GPT-4o on English, Task 3, for the one-shot with rationales setting
./run_benchmark.sh English 3 oneshot_with_rationales
```
To run all tasks for a specific language and setting, you can use a loop:

```
# Example: Run all 8 tasks for Japanese in the zero-shot no-rationales setting
for task in {1..8}; do
  ./run_benchmark.sh Japanese $task zeroshot_no_rationales
done
```

### 4. Running Evaluation
The run_evaluation.py script uses Gemini 1.5 Pro to score the generated outputs. Please note that this step can be costly (see our paper's Appendix for a full cost analysis).
```
# Example: Evaluate all outputs generated for the GPT-4o model on the English benchmark
python scripts/run_evaluation.py --model gpt-4o --language English
```

Note: For convenience, we provide our pre-computed model outputs and evaluation scores in the results/ directory, allowing you to bypass these computationally intensive steps.

---
# License
The code, annotations, and other original materials in this repository are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). The images are not covered by this license and are subject to their original copyright terms.
# Citation
If you use VLURes in your research, please cite our paper.
```
Under review at the Journal of Machine Learning Research
```

# Contact
For questions about the paper or repository, please contact the corresponding authors:
- Jesse Atuhurra (atuhurra.jesse.ag2@naist.ac.jp)
- Tatsuya Hiraoka (tatsuya.hiraoka@mbzuai.ac.ae)



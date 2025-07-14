# VLURes: Understanding Vision and Language Across Cultures

**A New Benchmark for Smarter, More Equitable AI**

*Jesse Atuhurra<sup>1</sup>, Iqra Ali<sup>2</sup>, Tomoya Iwakura<sup>3</sup>, Hidetaka Kamigaito<sup>1</sup>, and Tatsuya Hiraoka<sup>4</sup>*  
<sup>1</sup> NAIST <sup>2</sup> QMUL <sup>3</sup> Fujitsu Ltd <sup>4</sup> MBZUAI


<!-- [**Paper**](https://www.jmlr.org/papers/v23/21-0000.html) -->
[**Code**](https://github.com/jatuhurrra/VLURes/) | [**Data**](https://huggingface.co/datasets/atamiles/VLURes)

<!-- ++++++++++++++++ ++++++++ ++++++++ Motivation (( ORIGINAL MAGENTA HERE cc00aa ))  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🌍 Motivation: A Multilingual, Multimodal World Needs Multilingual, Multimodal AI
</div>

Despite recent advances in Vision-Language Models (VLMs), most benchmarks evaluate models in English, with limited regard for non-English languages or rich, real-world contexts. 
This monolingual bias severely limits how we assess AI’s true generalization capabilities, especially for low-resource languages.

**VLURes** is designed to change that. It rigorously evaluates visual and linguistic understanding across **English, Japanese, Swahili**, and **Urdu**, using diverse tasks, rich prose, and grounded cultural contexts.

![VLURes Task Overview](https://raw.githubusercontent.com/jatuhurrra/VLURes/main/assets/aINTRO.png)
*Figure 1: VLURes Task Overview*  

<div style="background-color:#ffe0f7; border-left: 5px solid #00cccc; padding: 1em; margin-bottom: 1em;">
We envision a world comprising generalist intelligent agents, such as robots, that accomplish several Vision-Language tasks.
</div>

<!-- ++++++++++++++++ ++++++++ ++++++++  What We Built: The VLURes Benchmark  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🌍 What We Built: The VLURes Benchmark
</div>

VLURes is more than just a dataset; it's a comprehensive testbed for the next generation of intelligent agents.

*   **Truly Multilingual:** We collected 1,000 culturally-relevant image-text pairs for each of four languages: **English, Japanese, Swahili, and Urdu.**
*   **Rich, Real-World Context:** Instead of short captions, each image is paired with a full article, forcing the AI to reason about deep, contextual information.
*   **A New Test of "Unrelatedness":** We introduce a novel task that challenges models to identify and ignore textual information that is *not* related to an image—a crucial skill for navigating noisy, real-world data.


<!-- ++++++++++++++++ ++++++++ ++++++++   What Is VLURes?  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧠 What Is VLURes?
</div>

**VLURes** is a multilingual vision-language benchmark aimed at testing intelligent agents under realistic conditions. Each input contains an image and an article-level text (not just captions), and the benchmark tests a model’s ability to perform both **image-only** and **image+text** reasoning.

VLURes covers 8 tasks:
- Object Recognition (OR)
- Scene Understanding (SU)
- Relation Understanding (RU)
- Semantic Segmentation (SS)
- Image Captioning (IC)
- Image-Text Matching (ITM)
- Visual Question Answering (VQA)
- Unrelatedness (newly introduced)

<!-- ++++++++++++++++ ++++++++ ++++++++  Dataset Construction  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🏗️ Dataset Construction
</div>

We collected articles and images from multiple web sources, including Wikipedia, Wikinews, blogs, and forums. The collection covers diverse topics such as animals, locations, food, buildings, and events.

- **Languages:** English (En), Japanese (Jp), Swahili (Sw), Urdu (Ur)
- **Image-Text Pairs:** 1000+ pairs per language
- **Rich Context:** Full-length articles, not just captions
- **Cultural Coverage:** Data sourced from native content in all four languages

We used **CLIP similarity scores** to align the most relevant image to each article. All data was cleaned manually, filtered for quality, and checked for NSFW or offensive content.

<!-- ++++++++++++++++ ++++++++ ++++++++  Dataset Construction  ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🎯 New Task: The "Unrelatedness" Challenge
</div>

The proposed Unrelatedness task. Left: The VLM inputs consist of two modalities, a pair of images and texts. The image undergoes a series of transformations in the vision encoder and connector, generating visual tokens that are ready for alignment in a shared embedding space. Similarly, a tokenizer tokenizes text, generating textual tokens. Textual and visual tokens are aligned in a shared embedding space and fed as input to the LLM. Right. The LLM uses its multimodal understanding to decide what textual information is relevant to different parts of the image. We see that the text painted green (marked with a cross sign) is directly related to the region of the image shown inside a green square box. That is, the text matches the image part shown in green. But in this task, we are interested in text unrelated to the image. Hence, yellow text (marked with a check sign) answers our Unrelatedness task.

![VLURes Task Overview](https://raw.githubusercontent.com/jatuhurrra/VLURes/main/assets/UnrelatednessTask.png)
*Figure 2:  Our proposed Unrelatedness Task*  

Unlike traditional matching tasks, **Unrelatedness** tests whether a model can identify *irrelevant* information. This is vital in noisy, multimodal environments like news feeds or social media.

> Can the model *ignore* text that does not describe or relate to the image?  
> This is the inverse of standard grounding tasks and pushes models to reason beyond associations.

<!-- ++++++++++++++++ ++++++++ ++++++++   Summary of the Benchmark Pipeline  ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  📊 Summary of the Benchmark Pipeline
</div>

1. **Task Definition**: 8 vision-language tasks
2. **Data Collection**: From native-language web sources
3. **Alignment**: Image selection via CLIP similarity
4. **Evaluation**: Via human and automatic judges
5. **Results**: Quantitative accuracy + qualitative rationale analysis

<!-- ++++++++++++++++ ++++++++ ++++++++   Evaluation Protocols  ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🔬 Evaluation Protocols
</div>

Models were tested under:
- **Zero-shot and One-shot settings**
- **With and without rationales**
- **Before and after fine-tuning**

We used both:
- **Automatic evaluation**: via Gemini 1.5 Pro ("LLM-as-a-Judge")
- **Human evaluation**: native speakers rated output quality on a 1–100 scale

![VLURes Task Performance](https://raw.githubusercontent.com/jatuhurrra/VLURes/main/assets/radarPlots.png)

<!-- ++++++++++++++++ ++++++++ ++++++++  Experiment Results: Key Findings ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧪 Experiment Results: Key Findings
</div>

- **GPT-4o** is the top performer across all settings but still trails human performance, especially for Swahili and Urdu.
- **Rationales help**: prompting models to “show their work” consistently improved accuracy.
- **Open-source models** like Qwen2VL and PALO significantly benefit from fine-tuning, but struggle with Swahili and Urdu input.
- **Multilingual drop**: performance degrades in the order En → Jp → Ur → Sw, showing clear signs of language bias.

<!-- ++++++++++++++++ ++++++++ ++++++++  Challenges Highlighted ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  📉 Challenges Highlighted
</div>

- **Poor Swahili/Urdu coverage** in even the strongest open-source models
- **Lack of robustness** in outputs when prompts and answers are in different languages
- **Language alignment** (En input + En output) still yields the best performance
- **Rationale prompting** significantly closes the gap between open-source and proprietary models

<!-- ++++++++++++++++ ++++++++ ++++++++   Open Access ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
   🔓 Open Access
</div>
We believe in open science. All assets are publicly available:

<!--
*   [**Read the Full Paper**](https://www.jmlr.org/papers/v23/21-0000.html)
-->

*   [**Explore the Dataset on Hugging Face**](https://huggingface.co/datasets/atamiles/VLURes)
*   [**View the Code on GitHub**](https://github.com/jatuhurrra/VLURes/)

<!-- ++++++++++++++++ ++++++++ ++++++++   Authors, BibTeX, Usage and License Notices ++++++++ ++++++++ ++++++++ --> 
<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
   🧑‍💻 Authors, BibTeX, Usage and License Notices
</div>

### 🧑‍💻 Authors
For questions about this research, please get in touch with the corresponding authors:

*   **Jesse Atuhurra** (`atuhurra.jesse.ag2@naist.ac.jp`)
*   **Tatsuya Hiraoka** (`tatsuya.hiraoka@mbzuai.ac.ae`)

### 📚 BibTeX

```
Under review at Journal of Machine Learning Research
```

### Usage and License Notices

The code, annotations, and other original materials in this repository are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

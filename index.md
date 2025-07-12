---
title: VLURes Project
---

# VLURes: Understanding Vision and Language Across Cultures

*A New Benchmark for Smarter, More Equitable AI*

[![Read the Paper](https://img.shields.io/badge/Read%20the%20Paper-JMLR-b31b1b.svg)](https://www.jmlr.org/papers/v23/21-0000.html)
[![View the Code](https://img.shields.io/badge/View%20the%20Code-GitHub-181717.svg?logo=github)](https://github.com/jatuhurrra/VLURes/)
[![Explore the Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/atamiles/VLURes)

---

## The Challenge: Does AI Understand the World in More Than Just English?

Today's most advanced AI can describe images and understand text with amazing accuracy. But most of these systems are trained and tested on English data. This leaves a critical question unanswered: how well do these models perform when faced with the diverse languages and cultural contexts that make up our world?

Our research addresses this gap. We created **VLURes**, a new benchmark designed to test how well AI models handle the complex interplay of images and long, detailed text in multiple languages, including low-resource languages that are often overlooked.

![VLURes Task Overview](https://raw.githubusercontent.com/jatuhurra/VLURes/main/path/to/your/figure1.png)

## What We Built: The VLURes Benchmark

VLURes is more than just a dataset; it's a comprehensive testbed for the next generation of intelligent agents.

*   **Truly Multilingual:** We collected 1,000 culturally-relevant image-text pairs for each of four languages: **English, Japanese, Swahili, and Urdu**.
*   **Rich, Real-World Context:** Instead of short captions, each image is paired with a full article, forcing the AI to reason about deep, contextual information.
*   **A New Test of "Unrelatedness":** We introduce a novel task that challenges models to identify and ignore textual information that is *not* related to an image—a crucial skill for navigating noisy, real-world data.

## Our Key Findings

Our evaluation of ten leading AI models revealed that even the most advanced systems struggle with multilingual understanding.
*   **A Persistent Gap:** Even the best model, GPT-4o, lags behind human performance, and this gap widens significantly for open-source models.
*   **The Low-Resource Challenge:** Performance drops dramatically for Swahili and Urdu, highlighting the urgent need for more equitable AI development.
*   **The Value of Rationale:** We found that prompting models to "show their work" by generating rationales consistently improves performance and transparency.

## Explore Our Work

We have made all our resources publicly available to encourage further research and development in this vital area.

*   **[Read the Full Paper](https://www.jmlr.org/papers/v23/21-0000.html)** for a deep dive into our methodology and results.
*   **[Explore the Dataset on Hugging Face](https://huggingface.co/datasets/atamiles/VLURes)** to see the image-text pairs yourself.
*   **[View the Code on GitHub](https://github.com/jatuhurrra/VLURes/)** to reproduce our experiments.

---

### Contact

For questions about this research, please contact the corresponding authors:
*   **Jesse Atuhurra** (`atuhurra.jesse.ag2@naist.ac.jp`)
*   **Tatsuya Hiraoka** (`tatsuya.hiraoka@mbzuai.ac.ae`)


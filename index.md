---
layout: default
title: VLURes Project
---

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">The Challenge: Does AI Understand the World in More Than Just English?</h2>
        <div class="content has-text-justified">
          <p>
            Today's most advanced AI can describe images and understand text with amazing accuracy. But most of these systems are trained and tested on English data. This leaves a critical question unanswered: how well do these models perform when faced with the diverse languages and cultural contexts that make up our world?
          </p>
           <p>
            Our research addresses this gap. We created <b>VLURes</b>, a new benchmark designed to test how well AI models handle the complex interplay of images and long, detailed text in multiple languages, including low-resource languages that are often overlooked.
          </p>
        </div>
      </div>
    </div>
    <div class="columns is-centered">
        <div class="column has-text-centered">
            <img src="https://raw.githubusercontent.com/jatuhurra/VLURes/main/path/to/your/figure1.png" alt="VLURes Task Overview" />
            <p>Figure 1: VLURes Task Overview</p>
        </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered">
      <div class="column is-full-width">
        <h2 class="title is-3">What We Built: The VLURes Benchmark</h2>
        <div class="content has-text-justified">
          <p>
            VLURes is more than just a dataset; it's a comprehensive testbed for the next generation of intelligent agents.
          </p>
          <ul>
            <li><b>Truly Multilingual:</b> We collected 1,000 culturally-relevant image-text pairs for each of four languages: <b>English, Japanese, Swahili, and Urdu.</b></li>
            <li><b>Rich, Real-World Context:</b> Instead of short captions, each image is paired with a full article, forcing the AI to reason about deep, contextual information.</li>
            <li><b>A New Test of "Unrelatedness":</b> We introduce a novel task that challenges models to identify and ignore textual information that is <em>not</em> related to an image—a crucial skill for navigating noisy, real-world data.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
    <div class="container is-max-desktop">
      <h2 class="title is-3">Our Key Findings</h2>
      <div class="content has-text-justified">
        <p>
            Our evaluation of ten leading AI models revealed that even the most advanced systems struggle with multilingual understanding.
        </p>
        <ul>
            <li><b>A Persistent Gap:</b> Even the best model, GPT-4o, lags behind human performance, and this gap widens significantly for open-source models.</li>
            <li><b>The Low-Resource Challenge:</b> Performance drops dramatically for Swahili and Urdu, highlighting the urgent need for more equitable AI development.</li>
            <li><b>The Value of Rationale:</b> We found that prompting models to "show their work" by generating rationales consistently improves performance and transparency.</li>
        </ul>
      </div>
    </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Explore Our Work</h2>
    <div class="content has-text-centered">
      <p>We have made all our resources publicly available to encourage further research and development in this vital area.</p>
       <div class="publication-links">
            <span class="link-block">
            <a href="https://www.jmlr.org/papers/v23/21-0000.html"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon"><i class="fas fa-file-pdf"></i></span>
                <span>Read the Full Paper</span>
            </a>
            </span>
            <span class="link-block">
            <a href="https://huggingface.co/datasets/atamiles/VLURes"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon"><i class="fas fa-database"></i></span>
                <span>Explore the Dataset on Hugging Face</span>
            </a>
            </span>
            <span class="link-block">
            <a href="https://github.com/jatuhurrra/VLURes/"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon"><i class="fab fa-github"></i></span>
                <span>View the Code on GitHub</span>
            </a>
            </span>
      </div>
    </div>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Contact</h2>
    <div class="content has-text-justified">
        <p>For questions about this research, please contact the corresponding authors:</p>
        <ul>
            <li><b>Jesse Atuhurra</b> (`atuhurra.jesse.ag2@naist.ac.jp`)</li>
            <li><b>Tatsuya Hiraoka</b> (`tatsuya.hiraoka@mbzuai.ac.ae`)</li>
        </ul>
    </div>
  </div>
</section>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{atuhurra2025vlures,
  title={VLURes: Understanding Vision and Language Across Cultures},
  author={Atuhurra, Jesse and Hiraoka, Tatsuya and and others},
  journal={Journal of Machine Learning Research},
  year={2025},
  url={https://www.jmlr.org/papers/v23/21-0000.html}
}</code></pre>
  </div>
</section>

## About

This repository contains the research and implementation artifacts for the master's thesis
**"Exploiting Artificial Intelligence in Software Development"** at Comenius University in Bratislava.

The thesis explores the use of large language models in the domain of **log anomaly detection**.
The goal is to design and develop an agentic AI system capable of analyzing log sequences and
producing both an **anomaly classification** and a **natural language explanation** of the
detected issue — moving beyond simple detection towards interpretable, human-readable diagnostics.

The ultimate objective is to deliver a **runnable, user-friendly framework** that can be applied
to real-world log data with minimal setup. Key design principles guiding the development are:

- **Generalizability** — the framework should perform well on unseen data and new environments
  without requiring extensive fine-tuning or retraining.
- **Robustness** — the system should maintain consistent performance even when the structure or
  format of incoming logs changes over time.
- **Accessibility** — the framework should be easy to run on custom data, with as little
  model fine-tuning required from the user as possible, ideally none at all.

These principles represent our **aspirational targets** rather than guaranteed outcomes. The aim
is to design the agentic system in a way that incorporates as many of these properties as
feasibly possible, to the best of our ability within the scope of the thesis.

The system will be evaluated and trained on publicly available log datasets and assessed using
both classification metrics and explanation quality benchmarks.

## Task Calendar

### ✅ Completed Tasks

| Week | Dates | Task |
|------|-------|------|
| Week 1 | Feb 16 – Feb 22 | Literature search — surveying existing research on log anomaly detection using LLMs and agentic AI approaches |
| Week 2 | Feb 23 – Mar 1 | Selection of top 5 most relevant papers based on recency, methodology, and applicability |
| Week 3 | Mar 2 – Mar 8 | In-depth reading and analysis of selected papers |
| Week 4 | Mar 9 – Mar 15 | In-depth reading and analysis of selected papers (continued) |
| Week 5 | Mar 16 – Mar 22 | Design decisions for the first prototype — selecting frameworks, models, and datasets based on findings from the papers |
| Week 6 | Mar 23 – Mar 29 | Implementation of the first prototype — a two-node agentic AI pipeline accepting log sequences and producing anomaly classification and cause explanation |
| Week 7 | Mar 30 – Apr 5 | Implementation of the first prototype (continued) and initial testing |
| Week 8 | Apr 6 – Apr 12 | Designing evaluation methodology — defining metrics for classification correctness and explanation quality |
| Week 9 | Apr 13 – Apr 19 | Running initial benchmarks on the prototype using HDFS, BGL, and Thunderbird datasets |
| Week 10 | Apr 20 – Apr 26 | Analysis of benchmark results and investigation of data scarcity problem — exploring data augmentation strategies |
| Week 11 | Apr 27 – May 3 | Design and implementation of LLM-based few-shot synthetic data augmentation script |
| Week 12 | May 4 – May 11 | Testing and validation of the data augmentation pipeline |
| Week 13 | May 12 – May 18 | Preparation of final seminar presentation, drafting initial structure of the thesis paper |

### 🗓️ Planned Tasks

| # | Task |
|---|------|
| 1 | Augment datasets using the implemented script and establish a validation strategy to ensure quality and correctness of augmented data |
| 2 | Explore and implement an alternative augmentation approach: extracting abnormal log messages from anomalous sequences and injecting them into normal sequences to create synthetic anomalies with known root causes |
| 3 | Extend the prototype to support tool use within the agentic AI framework |
| 4 | Design and implement dedicated tools for the agentic AI (e.g. log parsers, retrieval tools, classification backends) |
| 5 | Set up and run one of the LLM-based log anomaly detection frameworks identified in the literature review (e.g. LogLLM) |
| 6 | Integrate the selected detection framework as a tool within the agentic AI pipeline and evaluate the combined system |

## Links & Resources

### 📄 Research Papers
| Paper | Description |
|-------|-------------|
| [LogGPT](https://ieeexplore.ieee.org/document/10466820) | Log anomaly detection using GPT-based large language models |
| [LLM Prompt Patterns](https://dl.acm.org/doi/10.5555/3721041.3721046) | A catalog of prompt patterns for enhancing prompt engineering with ChatGPT |
| [LogRESP-Agent](https://doi.org/10.3390/app15137237) | Agentic approach to log-based anomaly detection and response |
| [LLM-LADE](https://doi.org/10.1016/j.knosys.2025.114064) | LLM-based log anomaly detection and explanation |
| [LogLLM](https://arxiv.org/abs/2405.14328) | Log anomaly detection using Llama and BERT |

### 🗂️ Datasets
| Resource | Description |
|----------|-------------|
| [LogHub](https://github.com/logpai/loghub) | Collection of publicly available log datasets (HDFS, BGL, Thunderbird, etc.) |
| [LLM-LADE Seed Data](https://github.com/sleep-zzw-bot/LLM-LADE/tree/master/seed_data) | Annotated log sequences from LogHub labeled by LLM-LADE |

### 💻 Implementation
| Resource | Description |
|----------|-------------|
| [Agentic AI Prototype](https://github.com/kristianrusnak/mt_logs/blob/main/agentic_ai/code/v1/agentic.py) | Source code of the first agentic AI pipeline prototype |
| [Benchmarking Code](https://github.com/kristianrusnak/mt_logs/tree/main/agentic_ai/benchmarks) | Scripts used to evaluate the prototype against log datasets |
| [Data Augmentation](https://github.com/kristianrusnak/mt_logs/tree/main/datasets/augmentation) | LLM-based few-shot synthetic data augmentation pipeline |

### 📊 Results
| Resource | Description |
|----------|-------------|
| [Prototype v1 Evaluation Results](https://github.com/kristianrusnak/mt_logs/tree/main/agentic_ai/prototype/v1) | Benchmark results from the first prototype evaluation — each dataset has its own subfolder |

### 📝 Thesis
| Resource | Description |
|----------|-------------|
| [Master Thesis Paper](https://github.com/kristianrusnak/mt_logs/tree/main/master_thesis) | Work-in-progress master thesis document |
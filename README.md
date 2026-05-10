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

### 🗓️ Planned Tasks

| # | Task |
|---|------|
| 1 | Augment datasets using the implemented script and establish a validation strategy to ensure quality and correctness of augmented data |
| 2 | Explore and implement an alternative augmentation approach: extracting abnormal log messages from anomalous sequences and injecting them into normal sequences to create synthetic anomalies with known root causes |
| 3 | Extend the prototype to support tool use within the agentic AI framework |
| 4 | Design and implement dedicated tools for the agentic AI (e.g. log parsers, retrieval tools, classification backends) |
| 5 | Set up and run one of the LLM-based log anomaly detection frameworks identified in the literature review (e.g. LogLLM) |
| 6 | Integrate the selected detection framework as a tool within the agentic AI pipeline and evaluate the combined system |
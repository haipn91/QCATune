# QCATune
*A fine-tuning framework based on question, context, and answer relationships for enhancing legal information retrieval*
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.engappai.2025.111570-blue)](https://doi.org/10.1016/j.engappai.2025.111570)

---
## 📄 About the Paper  
Legal document retrieval is a complex and essential task within the legal domain, requiring the extraction of relevant legal documents based on specific questions. The complexity of legal texts, along with the high level of comprehension required, poses significant challenges. These challenges are particularly pronounced in low-resource languages and specialized domains, where data scarcity and linguistic nuances impede effective retrieval. 

To address these issues, we introduce a fine-tuning framework based on the relationships between questions, context, and answers (QCATune). This framework proposes two approaches: the first is fine-tuning based on question-context and question–answer relationships, and the second extends this by also incorporating answer-context relationships.

This repository accompanies our article:

> **A fine-tuning framework based on question, context, and answer relationships for enhancing legal information retrieval**  
> _Engineering Applications of Artificial Intelligence_, 159 (2025) 111570  
> https://doi.org/10.1016/j.engappai.2025.111570

## Directory Structure

QCATune/

├── data_preparation/

├── Raw_data/

├── Synthetic_data_generation/

├── data_rag/

│ │ ├── vibilaw/

│ │ ├── coling2020/

│ │ ├── zalo2021/

├── results/

│ ├── results_vibilaw/

│ │ ├── model1_alpha_beta/

│ │ │ ├── epoch_1/

│ │ │ ├── epoch_2/

│ │ │ ├── ...

│ │ │ ├── epoch_5/

│ │ │ └── metrics.json

│ │ ├──......

│ ├── results_coling2020/

│ │ ├── model1_alpha_beta/

│ │ │ ├── epoch_1/

│ │ │ ├── ...

│ │ │ ├── epoch_5/

│ │ │ └── metrics.json

│ │ ├──......

│ ├── results_zalo2021/

│ │ ├── model1_alpha_beta/

│ │ │ ├── epoch_1/

│ │ │ ├── ...

│ │ │ ├── epoch_5/

│ │ │ └── metrics.json

│ │ ├──......

├── custom_loss.py

├── fine_tune_model_vibilaw.py

├── fine_tune_model_zalo2021.py

├── fine_tune_model_coling2020.py

└── README


## Fine-Tune Model

### Config for Fine-Tuning:

- **models**: List of models to train (e.g., `keepitreal/vietnamese-sbert`).
- **alphas**: List of alpha values to experiment with (e.g., `[0.2, 0.3, 0.4, 0.5]`).
- **betas**: List of beta values to experiment with (e.g., `[0.2, 0.3, 0.4, 0.5]`).
- **loss_option**: Set `"qc-qa"` for QC-QA finetuning, `"qc-qa-ac"` for QC-QA-AC finetuning, and `"qc"` for baseline fine-tuning.

### Fine-Tuning Results:

The models are saved in folders with the above structure in the `results` folder.

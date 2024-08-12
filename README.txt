# QCA Fine-Tuning Framework for Legal Document Retrieval

Legal document retrieval is a complex and essential task within the legal domain, requiring the extraction of relevant legal documents based on specific questions. The complexity of legal texts, along with the high level of comprehension required, poses significant challenges. These challenges are particularly pronounced in low-resource languages and specialized domains, where data scarcity and linguistic nuances impede effective retrieval. 

This repository introduces a novel framework, QCA (Question-Context-Answer) fine-tuning, which includes two approaches: QC-QA fine-tuning and QC-QA-AC fine-tuning. 

## Directory Structure

QAC_fine_tuning/
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
│ │ │ ├── epoch_3/
│ │ │ ├── epoch_4/
│ │ │ ├── epoch_5/
│ │ │ └── metrics.json
│ │ ├──......
│ ├── results_coling2020/
│ │ ├── model1_alpha_beta/
│ │ │ ├── epoch_1/
│ │ │ ├── epoch_2/
│ │ │ ├── epoch_3/
│ │ │ ├── epoch_4/
│ │ │ ├── epoch_5/
│ │ │ └── metrics.json
│ │ ├──......
│ ├── results_zalo2021/
│ │ ├── model1_alpha_beta/
│ │ │ ├── epoch_1/
│ │ │ ├── epoch_2/
│ │ │ ├── epoch_3/
│ │ │ ├── epoch_4/
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

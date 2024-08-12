import os
import warnings
import torch
import json
import random
from datasets import Dataset
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
from custom_loss import CustomMultipleNegativesRankingLoss
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

BATCH_SIZE = 16

# Define the best alpha and beta parameters for each model
model_params = {
    "keepitreal/vietnamese-sbert": (0.2, 0.2),
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base": (0.4, 0.5),
    "bkai-foundation-models/vietnamese-bi-encoder": (0.5, 0.4),
    "hmthanh/VietnamLegalText-SBERT": (0.5, 0.2),
    "BAAI/bge-base-en-v1.5": (0.5, 0.4),
    "BAAI/bge-small-en-v1.5": (0.3, 0.4),
    "colbert-ir/colbertv2.0": (0.5, 0.4),
    "FPTAI/vibert-base-cased": (0.4, 0.5),
    "vinai/phobert-large": (0.4, 0.2),
    "vinai/phobert-base": (0.3, 0.2)
}

def fine_tune(model_name, train_examples, val_dataset_file, output_dir, alpha, beta, batch_size=BATCH_SIZE, epochs=5, save_steps=1):
    
    model_dir = os.path.join(output_dir, f"{model_name.replace('/', '_')}_alpha_{alpha}_beta_{beta}")
    
    # Check if results already exist
    metrics_file = os.path.join(model_dir, "metrics.json")
    test_metrics_file = os.path.join(model_dir, "test_metrics.json")
    if os.path.exists(metrics_file) and os.path.exists(test_metrics_file):
        logging.info(f"Skipping training for {model_dir} as results already exist.")
        return

    model = SentenceTransformer(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=batch_size)
    train_loss = CustomMultipleNegativesRankingLoss(model=model, alpha=alpha, beta=beta, loss_option="qc-qa-ac")

    with open(val_dataset_file, 'r', encoding='utf-8') as f:
        val_dataset = json.load(f)   
    val_evaluator = InformationRetrievalEvaluator(
        val_dataset['queries'], 
        val_dataset['corpus'], 
        val_dataset['relevant_docs'],
        accuracy_at_k=[1, 3, 5, 10, 20],
        precision_recall_at_k=[1, 3, 5, 10, 20],
        mrr_at_k=[1, 3, 5],
        ndcg_at_k=[1, 5, 20],
        map_at_k=[100],
        name='validation'
    )
    
    with open(test_dataset_file, 'r', encoding='utf-8') as f:
        test_dataset = json.load(f)
    test_evaluator = InformationRetrievalEvaluator(
        test_dataset['queries'], 
        test_dataset['corpus'], 
        test_dataset['relevant_docs'],
        accuracy_at_k=[1, 3, 5, 10, 20],
        precision_recall_at_k=[1, 3, 5, 10, 20],
        mrr_at_k=[1, 3, 5],
        ndcg_at_k=[1, 5, 20],
        map_at_k=[100],
        name='test'
    )
    
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    metrics = []
    test_metrics = []
    
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logging.info(f"Epoch {epoch}/{epochs}")
        
        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=val_evaluator,
                epochs=1, 
                warmup_steps=warmup_steps,
                show_progress_bar=True
            )
            
            metric = val_evaluator(model)
            metrics.append(metric)
            test_metric = test_evaluator(model)
            test_metrics.append(test_metric)
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            with open(test_metrics_file, 'w', encoding='utf-8') as f:
                if isinstance(test_metrics[0], list):
                    test_metrics_dict = [{'test_cosine_map@100': val} for val in test_metrics]
                    json.dump(test_metrics_dict, f, ensure_ascii=False, indent=4)
                else:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                
            if epoch % save_steps == 0:
                epoch_dir = os.path.join(model_dir, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                logging.info(f"Saving model to {epoch_dir}")
                model.save(epoch_dir)

        except RuntimeError as e:
            logging.error(f"RuntimeError during training: {e}")
            break

def load_and_split_dataset(train_dataset_file, split_ratio):
    with open(train_dataset_file, 'r', encoding='utf-8') as f:
        train_dataset = json.load(f)
    
    all_examples = []
    for query_id, query in train_dataset['queries'].items():
        doc_id = train_dataset['relevant_docs'][query_id][0]
        context = train_dataset['corpus'][doc_id]
        complete_answer = train_dataset['answers'][query_id]
        all_examples.append(InputExample(texts=[query, context, complete_answer]))

    split_size = int(len(all_examples) * split_ratio)
    return random.sample(all_examples, split_size)

if __name__ == "__main__":
    dataset_splits = [0.2, 0.4, 0.6, 0.8]
    train_dataset_file = f'../data_rag/zalo/new_train_dataset.json'
    val_dataset_file = f'../data_rag/zalo/new_val_dataset.json'
    test_dataset_file = f'../data_rag/zalo/test_dataset.json'
  
    output_base_dir = f'results/results_zalo'

    for model_name, (alpha, beta) in model_params.items():
        for split_ratio in dataset_splits:
            train_examples = load_and_split_dataset(train_dataset_file, split_ratio)
            output_dir = os.path.join(output_base_dir, f"split_{int(split_ratio*100)}")
            fine_tune(
                model_name=model_name,
                train_examples=train_examples,
                val_dataset_file=val_dataset_file,
                output_dir=output_dir,
                alpha=alpha,
                beta=beta,
                save_steps=1 
            )

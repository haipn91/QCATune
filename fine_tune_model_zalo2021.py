import json
import torch
from datasets import Dataset
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
from custom_loss import CustomMultipleNegativesRankingLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

BATCH_SIZE = 16
def fine_tune(model_name, train_dataset_file, val_dataset_file, output_dir, alpha, beta, batch_size=BATCH_SIZE, epochs=5, save_steps=1):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    
    with open(train_dataset_file, 'r', encoding='utf-8') as f:
        train_dataset = json.load(f)
    
    train_examples = []
    for query_id, query in train_dataset['queries'].items():
        doc_id = train_dataset['relevant_docs'][query_id][0]
        context = train_dataset['corpus'][doc_id]
        complete_answer = train_dataset['answers'][query_id]
        train_examples.append(InputExample(texts=[query, context, complete_answer]))
    
    train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=batch_size)
    train_loss = CustomMultipleNegativesRankingLoss(model=model, alpha=alpha, beta=beta, loss_option="qc-qa-ac") # set loss_option="qc-qa" for QC-QA finetuning or "qc" for baseline finetuning

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
    
    model_dir = os.path.join(output_dir, f"{model_name.replace('/', '_')}_alpha_{alpha}_beta_{beta}")
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
            metrics_file = os.path.join(model_dir, "metrics.json")
            test_metrics_file = os.path.join(model_dir, "test_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            with open(test_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=4)
            if epoch % save_steps == 0:
                epoch_dir = os.path.join(model_dir, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                logging.info(f"Saving model to {epoch_dir}")
                model.save(epoch_dir)

        except RuntimeError as e:
            logging.error(f"RuntimeError during training: {e}")
            break

if __name__ == "__main__":
    models = ["keepitreal/vietnamese-sbert", 
                 "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                 "bkai-foundation-models/vietnamese-bi-encoder", 
                 "hmthanh/VietnamLegalText-SBERT",
                 "BAAI/bge-base-en-v1.5",
                 "BAAI/bge-small-en-v1.5",
                 "colbert-ir/colbertv2.0",
                 "FPTAI/vibert-base-cased", 
                 "vinai/phobert-large",
                 "vinai/phobert-base", 
                 ]

    #alphas = [round(x, 1) for x in list(np.arange(0.0, 1.1, 0.1))] # For QC-QA finetuning
    alphas = [0.2, 0.3, 0.4, 0.5]
    betas = [0.2, 0.3, 0.4, 0.5]

    train_dataset_file = f'data_rag/zalo2021/new_train_dataset.json'
    val_dataset_file = f'data_rag/zalo2021/new_val_dataset.json'
    test_dataset_file = f'data_rag/zalo2021/test_dataset.json'
  
    output_base_dir = f'results/results_zalo'

    for model_name in models:
        for alpha in alphas:
            for beta in betas:
                if alpha == 0.5 and beta == 0.5:
                    continue
                fine_tune(
                    model_name=model_name,
                    train_dataset_file=train_dataset_file,
                    val_dataset_file=val_dataset_file,
                    output_dir=output_base_dir,
                    alpha=alpha,
                    beta=beta,
                    save_steps=1 
                )

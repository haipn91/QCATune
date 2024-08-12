from typing import Iterable, Dict, Any, Optional
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import MultipleNegativesRankingLoss

class CustomMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim, 
                 alpha: float = 0.5, beta: float = 0.5, loss_option: str = "qc") -> None:
        super(CustomMultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.alpha = alpha
        self.beta = beta
        self.loss_option = loss_option
        self.mnr_loss = MultipleNegativesRankingLoss(model, scale, similarity_fct)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Optional[Tensor] = None) -> Tensor:
        
        query = sentence_features[0]
        context = sentence_features[1]

        loss_qc = self.mnr_loss([query, context], labels)

        total_loss = torch.tensor(0.0, requires_grad=True)
        if self.loss_option == "qc":
            total_loss = loss_qc
        elif self.loss_option in ["qc-qa", "qc-qa-ac"]:
            answer = sentence_features[2]
            loss_qa = self.mnr_loss([query, answer], labels)
            if self.loss_option == "qc-qa":
                total_loss = (self.alpha * loss_qc + (1-self.alpha) * loss_qa) 
            else:  
                loss_ac = self.mnr_loss([answer, context], labels)
                total_loss = (self.alpha * loss_qc + self.beta * loss_qa + (1- self.alpha - self.beta) * loss_ac) 

        return total_loss

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "alpha": self.alpha,
            "beta": self.beta,
            "loss_option": self.loss_option
        }

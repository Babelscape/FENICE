from typing import List

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from metric.utils.utils import chunks, split_into_sentences, distinct


class ClaimExtractor:
    def __init__(
        self,
        model_name: str = "alescire94/t5-base-summarization-claim-extractor",
        device: str = "cuda:0",
        batch_size: int = 70,
    ):
        self.device = device
        # load model from HF
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def process_batch(self, batch: List[str]) -> List[List[str]]:
        predictions = []
        batches = list(chunks(batch, self.batch_size))
        for b in tqdm(batches, desc="Extracting claims..."):
            tok_input = self.tokenizer.batch_encode_plus(
                b, return_tensors="pt", padding=True
            ).to(self.device)
            claims = self.model.generate(**tok_input)
            claims = self.tokenizer.batch_decode(claims, skip_special_tokens=True)
            claims = [split_into_sentences(c) for c in claims]
            claims = [distinct(c) for c in claims]
            predictions.extend(claims)
        return predictions

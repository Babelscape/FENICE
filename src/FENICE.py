from typing import List, Dict, Optional
import numpy as np
import torch.cuda
from tqdm import tqdm
from src.claim_extractor.claim_extractor import ClaimExtractor
from src.coreference_resolution.coreference_resolution import CoreferenceResolution
from src.nli.nli_aligner import NLIAligner
from src.utils.utils import split_into_paragraphs, split_into_sentences_batched


class FENICE:
    def __init__(
        self,
        use_coref: bool = False,
        num_sent_per_paragraph: int = 10,
        sliding_paragraphs=False,
        claim_extractor_batch_size: int = 256,
        coreference_batch_size: int = 1,
        nli_batch_size: int = 256,
        device: str = "cuda:0",
    ) -> None:
        self.num_sent_per_paragraph = num_sent_per_paragraph
        self.claim_extractor_batch_size = claim_extractor_batch_size
        self.coreference_batch_size = coreference_batch_size
        self.nli_batch_size = nli_batch_size
        self.sliding_paragraphs = sliding_paragraphs
        self.sentences_cache = {}
        self.coref_clusters_cache = {}
        self.claims_cache = {}
        self.alignments_cache = {}
        self.use_coref = use_coref
        self.coref_model = (
            CoreferenceResolution(load_model=False) if use_coref else None
        )
        self.device = device

    def _score(self, sample_id: int, document: str, summary: str):
        doc_id = self.get_id(sample_id, document)
        sentences_offsets = self.sentences_cache[doc_id]
        sentences = [s[0] for s in sentences_offsets]
        offsets = [(s[1], s[2]) for s in sentences_offsets]
        # paragraphs are sliding windows of {num_sent_per_paragraph} sentences
        paragraphs = split_into_paragraphs(
            sentences,
            self.num_sent_per_paragraph,
            sliding_paragraphs=self.sliding_paragraphs,
        )
        # claim extraction
        summary_id = self.get_id(sample_id, summary)
        summary_claims = self.claims_cache.get(summary_id, [summary])
        alignments = []
        for claim_id, claim in enumerate(summary_claims):
            # sentence-level alignment
            sentence_level_alignment = self.get_alignment(
                premises=sentences,
                hypothesis=claim,
                sample_id=sample_id,
                hypothesis_id=claim_id,
            )
            coref_alignment = None
            if self.use_coref:
                coref_clusters = self.coref_clusters_cache[doc_id]
                # modified versions of {aligned_sentence} obtained through coreference resolution
                coref_premises = self.coref_model.get_coref_versions(
                    sentence=sentence_level_alignment["premise"],
                    text=document,
                    sentences=sentences,
                    offsets=offsets,
                    clusters=coref_clusters,
                )
                if coref_premises:
                    coref_alignment = self.get_alignment(
                        sample_id=sample_id,
                        hypothesis_id=claim_id,
                        hypothesis=claim,
                        premises=coref_premises,
                        alignment_prefix="coref",
                    )
            paragraph_level_alignment = None
            if paragraphs:
                paragraph_level_alignment = self.get_alignment(
                    premises=paragraphs,
                    hypothesis=claim,
                    sample_id=sample_id,
                    hypothesis_id=claim_id,
                    alignment_prefix="par",
                )
            doc_level_alignment = None
            if len(paragraphs) > 1:
                doc_level_alignment = self.get_alignment(
                    hypothesis=claim,
                    premises=[document],
                    sample_id=sample_id,
                    hypothesis_id=claim_id,
                    alignment_prefix="doc",
                )
                doc_level_alignment["premise"] = "DOCUMENT"
            sample_alignments = [
                sentence_level_alignment,
                coref_alignment,
                paragraph_level_alignment,
                doc_level_alignment,
            ]
            alignment = self.max_alignment(sample_alignments)
            alignments.append(alignment)
        score = np.mean([al["score"] for al in alignments])
        return {"alignments": alignments, "score": score}

    def max_alignment(self, sample_alignments):
        sample_alignments = [s for s in sample_alignments if s is not None]
        alignment = max(sample_alignments, key=lambda x: x["score"])
        return alignment

    def get_alignment(
        self,
        premises,
        hypothesis,
        sample_id,
        hypothesis_id,
        alignment_prefix: Optional[str] = None,
    ):
        alignments_ids = []
        for prem_id in range(len(premises)):
            pair_id = self.get_alignment_id(
                sample_id=sample_id,
                premise_id=prem_id,
                hypothesis_id=hypothesis_id,
                premise=premises[prem_id],
                alignment_prefix=alignment_prefix,
            )
            alignments_ids.append(pair_id)
        loaded_alignment = self.load_alignment(
            alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
        )
        if loaded_alignment is None:
            all_pairs = [(x, hypothesis) for x in premises]
            self.cache_alignment(alignments_ids, all_pairs)
            alignment, _ = self.load_alignment(
                alignments_ids=alignments_ids, premises=premises, hypothesis=hypothesis
            )
        else:
            alignment, _ = loaded_alignment
        return alignment

    def score_batch(self, batch: List[Dict[str, str]]) -> List[Dict]:
        documents = [el["document"] for el in batch]
        summaries = [el["summary"] for el in batch]
        self.cache(documents, summaries)
        predictions = []
        for sample_id, (doc, summary) in tqdm(
            enumerate(zip(documents, summaries)),
            total=len(documents),
            desc="Computing FENICE...",
        ):
            predictions.append(self._score(sample_id, doc, summary))
        return predictions

    def cache(self, documents, summaries):
        self.cache_sentences(documents)
        if self.use_coref:
            self.cache_coref(documents)
        self.cache_claims(summaries)
        self.cache_alignments(documents, summaries)

    def cache_sentences(self, documents):
        all_sentences = split_into_sentences_batched(
            documents, batch_size=512, return_offsets=True
        )
        for i, sentences in enumerate(all_sentences):
            id = self.get_id(i, documents[i])
            self.sentences_cache[id] = sentences

    def get_docs_to_process(self, documents, cache):
        ids = [doc_id for doc_id in range(len(documents))]
        ids_to_process = [
            id for id in ids if self.get_id(id, documents[id]) not in cache
        ]
        doc_to_process = [documents[i] for i in ids_to_process]
        return doc_to_process, ids_to_process

    def get_id(self, sample_id: int, text: str, k_chars: int = 100):
        id = f"{sample_id}{text[:k_chars]}"
        return id

    # cache claim extraction outputs
    def cache_claims(self, summaries):
        claim_extractor = ClaimExtractor(
            batch_size=self.claim_extractor_batch_size, device=self.device
        )
        claims_predictions = claim_extractor.process_batch(summaries)
        for summ_id, claims in enumerate(claims_predictions):
            id = self.get_id(summ_id, summaries[summ_id])
            self.claims_cache[id] = claims
        del claim_extractor
        torch.cuda.empty_cache()

    # cache coreference resolution outputs
    def cache_coref(self, documents):
        coref_model = CoreferenceResolution(
            batch_size=self.coreference_batch_size, device=self.device
        )
        all_clusters = coref_model.get_clusters_batch(documents)
        for doc_id, clusters in enumerate(all_clusters):
            id = self.get_id(doc_id, documents[doc_id])
            self.coref_clusters_cache[id] = clusters
        del coref_model.model
        torch.cuda.empty_cache()

    def cache_alignments(self, documents: List[str], summaries: List[str]):
        alignments_ids, all_pairs = [], []
        self.nli_aligner = NLIAligner(
            batch_size=self.nli_batch_size, device=self.device
        )
        for sample_id in range(len(summaries)):
            summary_id = self.get_id(sample_id, summaries[sample_id])
            document_id = self.get_id(sample_id, documents[sample_id])
            claims = self.claims_cache[summary_id]
            sentences = [s[0] for s in self.sentences_cache[document_id]]
            paragraphs = split_into_paragraphs(
                sentences,
                self.num_sent_per_paragraph,
                sliding_paragraphs=self.sliding_paragraphs,
            )
            alignments_ids, all_pairs = self.compute_nli_pairs(
                alignments_ids, all_pairs, claims, sample_id, sentences
            )
            alignments_ids, all_pairs = self.compute_nli_pairs(
                alignments_ids, all_pairs, claims, sample_id, paragraphs, prefix="par"
            )
            alignments_ids, all_pairs = self.compute_nli_pairs(
                alignments_ids,
                all_pairs,
                claims,
                sample_id,
                [documents[sample_id]],
                prefix="doc",
            )
        self.cache_alignment(alignments_ids, all_pairs)

    def compute_nli_pairs(
        self, alignments_ids, all_pairs, claims, sample_id, premises, prefix=None
    ):
        for premise_id, premise in enumerate(premises):
            for claim_id, claim in enumerate(claims):
                alignment_id = self.get_alignment_id(
                    sample_id, premise_id, claim_id, premise, prefix
                )
                alignments_ids.append(alignment_id)
                all_pairs.append((premise, claim))
        return alignments_ids, all_pairs

    def get_alignment_id(
        self,
        sample_id: int,
        premise_id: int,
        hypothesis_id: int,
        premise: str,
        alignment_prefix: Optional[str] = None,
        k_chars: int = 100,
    ):

        id = f"{sample_id}-{premise_id}-{hypothesis_id}-{premise[:k_chars]}"
        if alignment_prefix is not None:
            id = f"{alignment_prefix}{id}"
        return id

    def cache_alignment(
        self, alignments_ids, all_pairs, disable_prog_bar: bool = False
    ):
        probabilities = self.nli_aligner.process_batch(
            all_pairs, disable_prog_bar=disable_prog_bar
        )
        for i, id in enumerate(alignments_ids):
            ent, contr, neut = (
                probabilities[0][i],
                probabilities[1][i],
                probabilities[2][i],
            )
            self.alignments_cache[id] = (ent, contr, neut)

    def load_alignment(
        self, alignments_ids: List[str], premises: List[str], hypothesis: str
    ):
        if all([k in self.alignments_cache for k in alignments_ids]):
            scores = [self.alignments_cache[key] for key in alignments_ids]
            alignment = None
            max_score = -np.inf
            for i, (ent, contr, neut) in enumerate(scores):
                ent, contr, neut = ent.item(), contr.item(), neut.item()
                align_score = ent - neut
                if align_score > max_score:
                    max_score = align_score
                    alignment = (premises[i], [ent, contr, neut])
            return (
                {
                    "premise": alignment[0],
                    "hypothesis": hypothesis,
                    "score": max_score,
                    "nli_dist": alignment[1],
                },
                scores,
            )
        else:
            return None

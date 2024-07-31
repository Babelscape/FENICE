from typing import List, Tuple
from fastcoref import FCoref
from tqdm import tqdm

from metric.utils.utils import chunks


class CoreferenceResolution:
    def __init__(
        self,
        input_max_toks: int = 10_000,
        pronouns_path: str = "data/pronouns.txt",
        batch_size: int = 1,
        load_model: bool = True,
        device: str = "cuda:0",
    ):
        self.input_max_toks = input_max_toks
        self.model = None
        if load_model:
            self.model = FCoref(device=device, enable_progress_bar=False)
        self.batch_size = batch_size
        with open(pronouns_path) as f:
            self.pronouns = f.read().splitlines()

    def get_clusters(self, document: str) -> List[List[Tuple[int, int]]]:
        # we truncate the document to a maximum length before executing coreference resolution
        # document = self.truncate_document(document)
        preds = self.model.predict(texts=[document])
        clusters = []
        if preds:
            clusters = preds[0].get_clusters(as_strings=False)
        return clusters

    # def truncate_document(self, document):
    #     return " ".join(document.split()[: self.input_max_toks])

    def get_clusters_batch(
        self, documents: List[str]
    ) -> List[List[List[Tuple[int, int]]]]:
        batched_documents = list(chunks(documents, n=self.batch_size))
        clusters = []
        for batch in tqdm(
            batched_documents,
            desc="Executing coreference resolution",
            total=len(batched_documents),
        ):
            # for i, doc in enumerate(batch):
            #     batch[i] = self.truncate_document(doc)
            batch_preds = self.model.predict(texts=batch)
            clusters.extend(
                [pred.get_clusters(as_strings=False) for pred in batch_preds]
            )
        return clusters

    # returns modified versions of {sentence} obtained through coreference resolution
    def get_coref_versions(
        self,
        sentence: str,
        text: str,
        sentences: List[str],
        offsets: List[Tuple[int, int]],
        clusters: List[List[Tuple[int, int]]],
    ):
        max_aligned_idx = [
            i for i, sent in enumerate(sentences) if sent.strip() == sentence.strip()
        ]
        if max_aligned_idx:
            max_aligned_idx = max_aligned_idx[0]
        else:
            return []
        document_sent_start, document_sent_end = offsets[max_aligned_idx]
        document_mentions = self.get_mentions(
            clusters, document_sent_start, document_sent_end
        )
        candidates = []
        for cluster in document_mentions:
            target_offsets = [
                (off[0], off[1])
                for off in cluster
                if document_sent_start <= off[0] and off[1] < document_sent_end
            ]
            target_words = [text[off[0] : off[1]] for off in target_offsets]
            relative_offsets = [
                (off[0] - document_sent_start, off[1] - document_sent_start)
                for off in target_offsets
            ]
            other_mentions = [
                text[ment[0] : ment[1]]
                for ment in cluster
                if text[ment[0] : ment[1]] not in target_words
            ]
            other_mentions = set(
                [m for m in other_mentions if m.lower() not in self.pronouns]
            )
            for target_word, target_offset in zip(target_words, relative_offsets):
                candidate = sentence[: target_offset[0]]
                for ment in other_mentions:
                    edited_candidate = sentence[target_offset[0] :].replace(
                        target_word, ment, 1
                    )
                    edited_candidate = candidate + edited_candidate
                    candidates.append(edited_candidate)
        return list(set(candidates))

    def get_mentions(
        self, clusters: List[List[Tuple[int, int]]], sent_start: int, sent_end: int
    ):
        return [
            cluster
            for cluster in clusters
            if any(
                [
                    (sent_start <= offset[0] and offset[1] < sent_end)
                    for offset in cluster
                ]
            )
        ]

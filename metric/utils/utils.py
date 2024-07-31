import hashlib
import pickle
from typing import List, Tuple, Union, Optional

import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def flatten(two_d_list: List[List]) -> List:
    """
    Flattens a 2D Python list into a 1D list using list comprehension.
    """
    return [item for sublist in two_d_list for item in sublist]


def chunks(lst: List, n: int) -> List[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def sliding_chunks(lst: List, n: int, sliding_stride: int = 1) -> List[List]:
    """Yield sliding windows of n-sized chunks from lst."""
    for i in range(len(lst) - n + 1):
        if i % sliding_stride == 0:
            yield lst[i : i + n]


def distinct(input_list: List) -> List:
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


def split_into_sentences(
    text: str, return_offsets: bool = False
) -> Union[List[Tuple[str, str, str]], List[str]]:
    # Process the text with spaCy
    doc = nlp(text)
    return get_sentences(doc, return_offsets)


def get_sentences(doc, return_offsets):
    # Initialize a list to store sentences and their offsets
    sentences_with_offsets = []
    # Iterate over the sentences in the processed text
    for sent in doc.sents:
        # Get the sentence text and character offsets
        sentence_text = sent.text
        start_offset = sent.start_char
        end_offset = sent.end_char

        # Add the sentence text and offsets to the list as a tuple
        sentences_with_offsets.append((sentence_text, start_offset, end_offset))
    if return_offsets:
        return sentences_with_offsets
    else:
        # Return only the sentences
        return [sentence[0] for sentence in sentences_with_offsets]


def split_into_sentences_batched(
    texts: List[str], return_offsets: bool = False, batch_size=32
) -> List[List[Tuple[str, str, str]]]:
    # Process the text with spaCy
    batches = list(chunks(texts, batch_size))
    sentences = []
    for b in tqdm(
        batches, total=len(batches), desc="splitting document batches into sentences"
    ):
        docs = nlp.pipe(b, disable=["attribute_ruler", "lemmatizer", "ner"])
        for doc in docs:
            doc_sentences = get_sentences(doc, return_offsets=return_offsets)
            sentences.append(doc_sentences)
    return sentences


def split_into_paragraphs(
    sentences: List[str],
    num_sent_per_paragraph: int,
    sliding_paragraphs=True,
    sliding_stride: int = 1,
) -> List[str]:
    if len(sentences) < num_sent_per_paragraph:
        return [" ".join(sentences)]
    paragraphs = (
        list(sliding_chunks(sentences, num_sent_per_paragraph, sliding_stride))
        if sliding_paragraphs
        else list(chunks(sentences, num_sent_per_paragraph))
    )
    for i, par in enumerate(paragraphs):
        paragraphs[i] = " ".join(par)
    return paragraphs


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_pickle(path: str):
    # Open the file in binary read mode
    with open(path, "rb") as file:
        # Load the object from the file
        return pickle.load(file)


def dump_pickle(path: str, data):
    # Open the file in binary write mode
    with open(path, "wb") as file:
        # Dump the object to the file
        pickle.dump(data, file)

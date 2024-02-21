import json
# import spacy
from spacy.lang.en import English
# import re


def print_json(data: dict | list | str) -> None:
    """
    Pretty prints the given dictionary or list.

    Args:
        data (dict | list): The data to pretty print.
    """
    if isinstance(data, str):
        data = json.loads(data)
    print(json.dumps(data, indent=4, sort_keys=True))


def split_into_sentences_spacy(text: str) -> list:
    """Split text into sentences using spaCy's Sentencizer."""
    nlp = English()  # Load the English tokenizer
    nlp.add_pipe('sentencizer')  # Add the sentencizer to the pipeline
    doc = nlp(text)  # Process the text
    sentences = [sent.text.strip() for sent in doc.sents]  # Get the sentences
    return sentences
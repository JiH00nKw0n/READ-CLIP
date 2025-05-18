import random
from typing import List

import nltk
import numpy as np
import spacy
import spacy
import torch
from nltk import word_tokenize
from spacy.matcher import Matcher
from spacy.tokens import Span, Token
from transformers.utils import is_torch_fx_proxy

from src.utils import pre_caption

nltk.download('punkt')
nltk.download('punkt_tab')

ALLOWED_POS_TAG = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "NUM",
    "PRON",
    "PROPN",
    "PART",
]


def swap_spans(tokens: List[Token], span1: Span, span2: Span) -> List[Token]:
    """
    Swap the positions of two spans in the list of tokens.

    Args:
        tokens (List[Token]): List of tokens in the document.
        span1 (Span): The first span to swap.
        span2 (Span): The second span to swap.

    Returns:
        List[Token]: The modified list of tokens with the spans swapped.
    """
    # Ensure span1 starts before span2
    if span2.start < span1.start:
        span1, span2 = span2, span1

    first_start = span1.start
    first_end = span1.end
    second_start = span2.start
    second_end = span2.end

    return (
            tokens[:first_start] +
            tokens[second_start:second_end] +
            tokens[first_end:second_start] +
            tokens[first_start:first_end] +
            tokens[second_end:]
    )


class NegCLIPNegativeTextMining:
    """
    A class for generating negative captions by swapping nouns, adjectives, adverbs,
    verb phrases, and noun phrases (only if composed of three or more tokens).

    Methods:
    --------
    generate_negative_captions(text: str) -> List[str]:
        Generates up to 5 negative captions by swapping elements in the caption.

    generate_sampled_negative_caption(text: str) -> str:
        Samples one negative caption from the generated set for training.
    """

    def __init__(self, max_num_texts: int = 5, rng=None):
        """
        Initializes the class with the maximum number of negative captions to generate
        and a random number generator with an optional seed.

        Args:
            max_num_texts (int, optional): The maximum number of negative captions to generate. Defaults to 5.
            rng (optional): Numpy random generator for reproducibility. If None, uses default seed 2024.
        """
        self.max_num_texts = max_num_texts
        self.rng = rng if rng else np.random.default_rng(2024)

        # spaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")

        # Verb phrase matcher
        self.vp_matcher = Matcher(self.nlp.vocab)
        self.vp_matcher.add(
            "Verb phrase", [
                [
                    {"POS": "VERB", "OP": "?"},
                    {"POS": "ADV", "OP": "*"},
                    {"POS": "AUX", "OP": "*"},
                    {"POS": "VERB", "OP": "+"}
                ]
            ]
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = 0
        pad_token_id = 0

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def generate_negative_captions(self, original_text: str) -> List[str]:
        """
        Generates up to 5 negative captions by swapping elements in the caption.

        Args:
            original_text (str): The input text as a string.

        Returns:
            List[str]: A list of generated negative captions.
        """
        doc = self.nlp(original_text)

        # Extract elements for swapping
        noun_phrases = [np for np in doc.noun_chunks if len(np) >= 3]  # Noun phrases with 3+ tokens
        verb_phrases = self.vp_matcher(doc, as_spans=True)
        adjectives = [doc[token.i:token.i + 1] for token in doc if token.pos_ == "ADJ"]
        adverbs = [doc[token.i:token.i + 1] for token in doc if token.pos_ == "ADV"]
        nouns = [doc[token.i:token.i + 1] for token in doc if token.pos_ == "NOUN"]

        # Define swap groups
        swap_groups = [
            ('NOUN', nouns),
            ('ADJ', adjectives),
            ('ADV', adverbs),
            ('NP', noun_phrases),
            ('VP', verb_phrases),
        ]

        negative_captions = []

        for _ in range(self.max_num_texts):
            eligible_groups = [(name, group) for name, group in swap_groups if len(group) >= 2]
            if not eligible_groups:
                break

            # Randomly choose a group and swap elements
            group_idx = self.rng.choice(len(eligible_groups))
            _, group = eligible_groups[group_idx]

            span_indices = self.rng.choice(len(group), size=2, replace=False)
            span1 = group[span_indices[0]]
            span2 = group[span_indices[1]]

            new_tokens = list(doc)
            new_tokens = swap_spans(new_tokens, span1, span2)

            # Join tokens into a string
            new_caption = " ".join([token.text for token in new_tokens])
            if new_caption not in negative_captions and new_caption != original_text:
                negative_captions.append(new_caption)

        return negative_captions

    def negative_caption(self, original_text: str) -> str:
        """
        Samples one negative caption from the generated set for training.

        Args:
            original_text (str): The input text as a string.

        Returns:
            str: A randomly selected negative caption, or the original text if no valid swaps.
        """
        negative_captions = self.generate_negative_captions(original_text)
        if negative_captions:
            return self.rng.choice(negative_captions)
        return ''


# Adapted from https://github.com/mertyg/vision-language-models-are-bows/blob/main/dataset_zoo/perturbations.py
class TextShufflerMixin:
    """
    A mixin class for text perturbation. Provides methods to shuffle or modify text
    based on various linguistic properties.
    """

    def __init__(self, seed: int = 2024):
        self.nlp = spacy.load("en_core_web_sm")
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def _get_token_indices(self, doc, tags: List[str]) -> List[int]:
        """
        Helper method to retrieve token indices for specific POS tags.

        Args:
            doc: Spacy Doc object.
            tags: List of POS tags to match.

        Returns:
            List of indices corresponding to the tokens with the specified tags.
        """
        return [i for i, token in enumerate(doc) if token.tag_ in tags]

    def shuffle_nouns_and_adj(self, text: str) -> str:
        """
        Shuffle nouns and adjectives in the text while keeping other tokens intact.

        Args:
            text: The input text to shuffle.

        Returns:
            The perturbed text with shuffled nouns and adjectives.
        """
        doc = self.nlp(text)
        tokens = np.array([token.text for token in doc])
        noun_idx = self._get_token_indices(doc, ['NN', 'NNS', 'NNP', 'NNPS'])
        adj_idx = self._get_token_indices(doc, ['JJ', 'JJR', 'JJS'])

        # Shuffle nouns and adjectives using default_rng
        tokens[noun_idx] = self.rng.permutation(tokens[noun_idx])
        tokens[adj_idx] = self.rng.permutation(tokens[adj_idx])

        return " ".join(tokens)

    def shuffle_all_words(self, text: str) -> str:
        """
        Shuffle all words in the text.

        Args:
            text: The input text to shuffle.

        Returns:
            The perturbed text with all words shuffled.
        """
        tokens = text.split()
        shuffled_tokens = self.rng.permutation(tokens)
        return " ".join(shuffled_tokens)

    def shuffle_allbut_nouns_and_adj(self, text: str) -> str:
        """
        Shuffle all words except nouns and adjectives.

        Args:
            text: The input text to shuffle.

        Returns:
            The perturbed text with shuffled words except nouns and adjectives.
        """
        doc = self.nlp(text)
        tokens = np.array([token.text for token in doc])
        noun_adj_idx = self._get_token_indices(doc, ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'])
        other_idx = np.ones(len(tokens), dtype=bool)
        other_idx[noun_adj_idx] = False

        tokens[other_idx] = self.rng.permutation(tokens[other_idx])
        return " ".join(tokens)

    def _get_trigrams(self, tokens: List[str]) -> List[List[str]]:
        """
        Split a list of tokens into trigrams.

        Args:
            tokens: List of tokens.

        Returns:
            List of trigrams.
        """
        return [tokens[i:i + 3] for i in range(0, len(tokens), 3)]

    def _shuffle_trigram(self, trigram: List[str]) -> List[str]:
        """
        Shuffle the words within a single trigram.

        Args:
            trigram: List of words in a trigram.

        Returns:
            Shuffled trigram.
        """
        return self.rng.permutation(trigram).tolist()

    def shuffle_within_trigrams(self, text: str) -> str:
        """
        Shuffle words within each trigram of the text.

        Args:
            text: The input text to shuffle.

        Returns:
            The perturbed text with shuffled trigrams.
        """
        tokens = word_tokenize(text)
        trigrams = self._get_trigrams(tokens)
        shuffled_trigrams = [self._shuffle_trigram(trigram) for trigram in trigrams]
        return " ".join([" ".join(trigram) for trigram in shuffled_trigrams])

    def shuffle_trigrams(self, text: str) -> str:
        """
        Shuffle the order of trigrams in the text.

        Args:
            text: The input text to shuffle.

        Returns:
            The perturbed text with shuffled trigram order.
        """
        tokens = word_tokenize(text)  # Tokenize the text
        trigrams = self._get_trigrams(tokens)  # Get trigrams
        self.py_rng.shuffle(trigrams)  # Shuffle the trigram order
        return " ".join([" ".join(trigram) for trigram in trigrams])

    def get_perturbed_sentences(self, text: str, max_words: int = 30) -> List[str]:
        """
        Generate a list of perturbed sentences using different perturbation methods.

        Args:
            text: The input text to perturb.
            max_words: Maximum number of words to include in the output sentences.

        Returns:
            A list of perturbed sentences.
        """
        perturb_functions = [
            self.shuffle_nouns_and_adj,
            self.shuffle_all_words,
            self.shuffle_allbut_nouns_and_adj,
            self.shuffle_within_trigrams,
            self.shuffle_trigrams,
        ]

        return [pre_caption(fn(text), max_words) for fn in perturb_functions]

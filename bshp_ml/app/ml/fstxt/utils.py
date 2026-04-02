import re

import pandas as pd

RE_CHARS = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")
RE_SPACES = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"[^a-zA-Zа-яА-ЯёЁ\s]")
RE_SMALL = re.compile(r"\b\w{1,2}\b")
RE_NUMBERS_EXCEPT_YEARS = re.compile(r"(?<!\S)(?!2\d{3}\b)\d+\b")


def preprocess_text(s: pd.Series) -> pd.Series:
    """
    Функция для очистки текста
    """
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(RE_CHARS, " ", regex=True)
    s = s.str.replace(RE_NUMBERS_EXCEPT_YEARS, "", regex=True)
    s = s.str.replace(RE_SPACES, " ", regex=True).str.strip()
    s = s.str.replace(RE_SMALL, " ", regex=True)

    return s


def prepare_sentences(df: pd.DataFrame, txt_cols) -> list[list[str]]:
    """
    Подготавливает предложения для обучения FastText
    """
    df_txt = df[txt_cols].astype(str)

    # HIGH_IMP = [
    #     "article_name",
    #     "payment_purpose",
    #     "payment_purpose_returned",
    #     "analytic",
    #     "analytic2",
    #     "analytic3",
    # ]
    # for col in HIGH_IMP:
    #     if col in df_txt.columns:
    #         # if col == "payment_purpose" or col == "payment_purpose_returned":
    #             # df_txt[col] = df_txt[col].str.replace(RE_NUMBERS, " ", regex=True)
    #         df_txt[col] = df_txt[col] + " "

    # df_txt = df_txt[HIGH_IMP].agg(" ".join, axis=1)
    df_txt = df_txt.agg(" ".join, axis=1)

    cleaned = preprocess_text(df_txt)

    sentences = cleaned.str.split().to_list()

    return sentences


def map_code_to_name():
    pass
    # det_map = {name: num for num, name in enumerate(df[col].unique())}


def zero_below_nth_max(df, cols, treshold=0.8, n=10):
    """
    Занулить значения, меньшие чем n-е максимальное значение в каждой колонке
    """

    def row_max(row):
        row.nlargest(n).iloc[-1]

    df_result = df.copy()
    df_result[cols] = df_result[cols].where(df_result[cols] >= treshold, 0)

    return df_result


def build_class_vocab(all_classes_names: dict, max_class_freq: float = 0.1) -> set[str]:
    """
    Build a set of distinctive tokens from class label names.
    Tokens appearing in more than max_class_freq fraction of class labels
    are considered generic (like "прочие", "иные", "расходы") and excluded.
    This way only discriminative tokens survive — ones that actually
    distinguish one class from another.
    """
    from collections import Counter

    all_names = []
    for class_list in all_classes_names.values():
        all_names.extend(class_list)

    total = len(all_names)
    token_doc_count: Counter = Counter()

    token_sets = []
    for name in all_names:
        tokens = set(preprocess_text(pd.Series([name])).iloc[0].split())
        tokens.discard("")
        token_sets.append(tokens)
        token_doc_count.update(tokens)

    vocab = {
        token
        for token, count in token_doc_count.items()
        if count / total <= max_class_freq
    }
    return vocab


def prepare_sentences_weighted(
    df: pd.DataFrame,
    article_cols: list[str],
    payment_cols: list[str],
    class_vocab: set[str],
    payment_weight: int = 2,
    word_weight_multipliers: dict[str, float] | None = None,
) -> list[list[str]]:
    """
    Prepares sentences with improvements:
    - payment_* tokens are filtered to class_vocab only (noise suppression)
      then repeated payment_weight times (signal amplification)
    - article_* tokens are kept as-is (already relatively clean)
    - word_weight_multipliers applies custom weights to specific words
      e.g. {'Оплата': 0.3, 'Зерноотходы': 0.6} reduces those words' importance
    """
    if word_weight_multipliers is None:
        word_weight_multipliers = {
            "оплата": 0.3,
            "зерноотходы": 0.6,
            "расходы": 0.5,
        }

    # Preprocess word multipliers to handle different cases
    word_weight_multipliers = {
        preprocess_text(pd.Series([k])).iloc[0]: v
        for k, v in word_weight_multipliers.items()
    }

    present_article = [c for c in article_cols if c in df.columns]
    present_payment = [c for c in payment_cols if c in df.columns]

    # Article side: plain preprocessing
    if present_article:
        art_text = df[present_article].astype(str).agg(" ".join, axis=1)
        art_tokens = preprocess_text(art_text).str.split().tolist()
    else:
        art_tokens = [[] for _ in range(len(df))]

    # Payment side: filter to class vocab only, then repeat
    if present_payment:
        pp_text = df[present_payment].astype(str).agg(" ".join, axis=1)
        pp_cleaned = preprocess_text(pp_text).str.split()
        pp_tokens = [
            _apply_word_weights(
                [t for t in tokens if t in class_vocab],
                payment_weight,
                word_weight_multipliers,
            )
            for tokens in pp_cleaned
        ]
    else:
        pp_tokens = [[] for _ in range(len(df))]

    sentences = [a + p for a, p in zip(art_tokens, pp_tokens)]
    return sentences


def _apply_word_weights(
    tokens: list[str], base_weight: int, word_weight_multipliers: dict[str, float]
) -> list[str]:
    """
    Apply custom weight multipliers to tokens.
    For each token, repeat it base_weight * multiplier times.
    If no multiplier is set, use base_weight repetitions.
    """
    result = []
    for token in tokens:
        multiplier = word_weight_multipliers.get(token, 1.0)
        count = max(1, int(base_weight * multiplier))
        result.extend([token] * count)
    return result

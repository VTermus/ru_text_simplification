import pandas as pd
#import json
#import requests
from typing import Dict, List, Optional, Tuple, Union
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# new
# helper for cefr conversion
def bucket(lbl: str) -> str:
    if lbl in ("A1", "A2"):
        return "A"
    if lbl in ("B1", "B2"):
        return "B"
    if lbl in ("C1", "C2"):
        return "C"
    raise ValueError(f"Unknown CEFR label: {lbl}")


# new
# helper for cefr conversion
def split_ABC(ts, ls):
    A, B, C = [], [], []
    for t, l in zip(ts, ls):
        b = bucket(l)
        if b == "A":
            A.append(t)
        elif b == "B":
            B.append(t)
        else:
            C.append(t)
    return A, B, C


# unchanged
def get_preference_pairs(priority_list: List[str],
                        other_list_1: List[str],
                        other_list_2: List[str]) -> Tuple[List[str], List[str]]:
    """Create preference pairs for reward modeling."""
    chosen = []
    rejected = []

    # Add samples from priority list as chosen samples
    for sample in priority_list:
        chosen.append(sample)

    # Add samples from other lists as rejected samples
    for sample in other_list_1:
        rejected.append(sample)
    for sample in other_list_2:
        rejected.append(sample)

    # Ensure equal number of chosen/rejected pairs
    rejected = shuffle(rejected, random_state=0)
    if len(chosen) < len(rejected):
        rejected = rejected[:len(chosen)]
    else:
        rejected = [rejected[i % len(rejected)] for i in range(len(chosen))]

    return chosen, rejected


# changed
def load_cefr_data(level: str, mode: str = "reward") -> Union[Dict[str, Dict[str, List[str]]], List[str]]:
    """Load and prepare CEFR data for training.

    Args:
        level: Target CEFR level (A, B, or C)
        mode: Either "reward" for reward modeling or "rl" for RL training

    Returns:
        For reward mode: Dict with train/eval data containing chosen/rejected pairs
        For RL mode: List of training sentences
    """
    # changed
    # load CEFR sentences
    df = pd.read_csv("CEFR_level_sentences.csv", sep=";", encoding="cp1251")
    df = df[["fragment", "textbook-assigned cefr level"]].dropna()
    df["fragment"] = df["fragment"].astype(str).str.strip()
    df["textbook-assigned cefr level"] = df["textbook-assigned cefr level"].astype(str).str.strip().str.upper()
    df = df[df["fragment"] != ""]

    texts = df["fragment"].tolist()
    labels = df["textbook-assigned cefr level"].tolist()

    # changed
    # train dev test split 80/10/10
    try:
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=0, stratify=labels
        )
        dev_texts, eval_texts, dev_labels, eval_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=0, stratify=temp_labels
        )
    except Exception:
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=0
        )
        dev_texts, eval_texts, dev_labels, eval_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=0
        )

    # changed
    train_A, train_B, train_C = split_ABC(train_texts, train_labels)
    dev_A, dev_B, dev_C = split_ABC(dev_texts, dev_labels)
    test_A, test_B, test_C = split_ABC(eval_texts, eval_labels)


    if mode == "reward":
        # Prepare preference pairs based on target level
        if level == "A":
            train_chosen, train_rejected = get_preference_pairs(train_A, train_B, train_C)
            dev_chosen, dev_rejected = get_preference_pairs(dev_A, dev_B, dev_C)
            eval_chosen, eval_rejected = get_preference_pairs(test_A, test_B, test_C)
        elif level == "B":
            train_chosen, train_rejected = get_preference_pairs(train_B, train_C, train_A)
            dev_chosen, dev_rejected = get_preference_pairs(dev_B, dev_C, dev_A)
            eval_chosen, eval_rejected = get_preference_pairs(test_B, test_C, test_A)
        else:  # level C
            train_chosen, train_rejected = get_preference_pairs(train_C, train_B, train_A)
            dev_chosen, dev_rejected = get_preference_pairs(dev_C, dev_B, dev_A)
            eval_chosen, eval_rejected = get_preference_pairs(test_C, test_B, test_A)

        return {
            "train": {"chosen": train_chosen, "rejected": train_rejected},
            "dev": {"chosen": dev_chosen, "rejected": dev_rejected},
            "eval": {"chosen": eval_chosen, "rejected": eval_rejected}
        }

    elif mode == "rl":  # mode == "rl"
        # For RL training, return sentences of target level
        if level == "A":
            return train_A
        elif level == "B":
            return train_B
        else:
            return train_C
    else:
        raise ValueError(f"Invalid mode: {mode}")


# changed (all)
def read_complicated_lines(path: str,
                          *,
                          min_len: int = 1,
                          max_cos_sim: Optional[float] = None) -> List[str]:
    """
    Reads csv with source and target sentences 
    
    Returns:
        List[str] of complex source sentences (RL inputs)

    Args:
        path: path to source_target_sentences.csv
        min_len: drop sources shorter than this many characters 
        max_cos_sim: keep only rows with cos_sim <= max_cos_sim
                     (to avoid identical source/target)
    """
    df = pd.read_csv(path, sep=";", encoding="cp1251")

    if "source" not in df.columns:
        raise ValueError(f"Expected column 'source' in {path}, got columns: {list(df.columns)}")

    src = df["source"].dropna().astype(str).str.strip()
    src = src[src != ""]

    # optional quality filters 
    if min_len > 1:
        src = src[src.str.len() >= min_len]

    if max_cos_sim is not None:
        if "cos_sim" not in df.columns:
            raise ValueError("max_cos_sim was set but column 'cos_sim' is missing in the CSV")
        cos = pd.to_numeric(df.loc[src.index, "cos_sim"], errors="coerce")
        src = src[cos <= max_cos_sim]

    return src.tolist()


# changed (all)
def get_complicated_sentence(path: str,
                             *,
                             min_len: int = 1,
                             max_cos_sim: Optional[float] = None) -> List[str]:
    """
    Returns the same thing as read_complicated_lines().
    """
    return read_complicated_lines(path, min_len=min_len, max_cos_sim=max_cos_sim)


'''
def read_complicated_lines(path: str) -> List[str]:
    """Reads GPT generated complicated sentences and extracts sentence list."""
    # with open(path, "r") as f:
    #     return json.load(f)["lines"]
    complicated_score_wiki = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            sent = json.loads(line)['response']['body']['choices'][0]['message']['content']
            complicated_score_wiki.append(sent)
    return complicated_score_wiki
'''
'''
def download_cefr_data(data_urls: Dict[str, str]) -> Dict[str, List[str]]:
    """Download data from URLs and split into lines."""
    data = {}
    for key, url in data_urls.items():
        response = requests.get(url)
        data[key] = response.text.strip().split("\n")
    return data

def get_level_sentences(samples: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Sort sentences into A, B, C levels."""
    level_A, level_B, level_C = [], [], []

    for sample in samples:
        s, l1, l2 = sample.split("\t")
        l1, l2 = int(l1), int(l2)

        if l1 == l2:
            if l1 <= 2:
                level_A.append(s)
            elif l1 <= 4:
                level_B.append(s)
            else:
                level_C.append(s)
        else:
            max_level = max(l1, l2)
            min_level = min(l1, l2)
            if max_level <= 2:
                level_A.append(s)
            elif min_level >= 3 and max_level <= 4:
                level_B.append(s)
            elif min_level >= 5:
                level_C.append(s)
            else:
                # For ambiguous cases, assign to higher level
                if max_level <= 4:
                    level_B.append(s)
                else:
                    level_C.append(s)

    return level_A, level_B, level_C
'''
'''
def get_complicated_sentence(path: str) -> List[str]:
    """Get complicated sentences from file."""
    complicated_score_wiki = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            sent = json.loads(line)['response']['body']['choices'][0]['message']['content']
            complicated_score_wiki.append(sent)
    return complicated_score_wiki
'''

from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import re


@dataclass
class Token:
    sent_id: str
    index: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str
    text: str
    translation: Optional[str] = None
    translation_en: Optional[str] = None
    title: Optional[str] = None

    def __post_init__(self):
        self.index = int(self.index) # coerce to int
        self.head = int(self.head)

def buildTokenList(myFile):
    annotation_keys = ['index', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
    sentences = []
    for line in myFile:
        if line.startswith('# text ='):
            currentToken = {}
            currentToken['text'] = line.split('=')[1].strip()
        elif line.startswith('# translation ='):
            currentToken['translation'] = line.split('=')[1].strip()
        elif line.startswith('# translation_en ='):
            currentToken['translation_en'] = line.split('=')[1].strip()
        elif line.startswith('# newdoc id ='):
            currentToken['title'] = line.split('=')[1].strip()
        elif line.startswith('# sent_id ='):
            currentToken['sent_id'] = line.split('=')[1].strip()
        elif re.match(r'\d+', line):
            currentToken.update(dict(zip(annotation_keys, re.split(r'\t+', line))))
            sentences.append(Token(**currentToken))
    return sentences

_TREE = {"df": None, "groups": None, "sent_col": None}

def init_tree_viewer(tokens_df, sent_col=None):
    """
    Standard setup. Run this once with your full dataframe.
    """
    global _TREE

    if sent_col is None:
        if "sent_id" in tokens_df.columns:
            sent_col = "sent_id"
        elif "sentid" in tokens_df.columns:
            sent_col = "sentid"
        else:
            raise ValueError("No sentence id column found.")

    df = tokens_df.copy()
    df["_sent_key"] = df[sent_col].astype(str)

    _TREE["df"] = df
    _TREE["groups"] = df.groupby("_sent_key", sort=False)
    _TREE["sent_col"] = sent_col

def print_clause(sent_id, head_index,
                 index_col="index", head_col="head", form_col="form",
                 upos_col="upos", deprel_col="deprel"):
    """
    Prints the clause headed by `head_index` in `sent_id` with full sentence metadata.
    The output is sorted by token index (linear order).
    """
    if _TREE["groups"] is None:
        raise RuntimeError("Run init_tree_viewer(tokensDf) once first.")

    # 1. Retrieve the sentence
    key = str(sent_id)
    try:
        s = _TREE["groups"].get_group(key).copy()
    except KeyError:
        # Fallback logic for int/float ID mismatch
        alts = [str(float(sent_id)), str(int(float(sent_id)))] if str(sent_id).replace('.', '').isdigit() else []
        for k2 in alts:
            if k2 in _TREE["groups"].groups:
                s = _TREE["groups"].get_group(k2).copy()
                break
        else:
            print(f"Sentence ID {sent_id} not found.")
            return

    start_node = int(head_index)
    
    # --- METADATA PRINTING ---
    print(f"Sentence: {sent_id} | Clause Head: {start_node}")
    if "text" in s.columns:
         print(f"Text:       {s['text'].iloc[0]}")
    if "translation" in s.columns:
        print(f"Mod. Fr.:   {s['translation'].iloc[0]}")
    if "translation_en" in s.columns:
        print(f"English:    {s['translation_en'].iloc[0]}")
    print("-" * 40)

    # 2. Data Cleaning
    s = s[pd.to_numeric(s[index_col], errors="coerce").notna()].copy()
    s[index_col] = s[index_col].astype(int)
    s[head_col] = pd.to_numeric(s[head_col], errors="coerce").fillna(0).astype(int)

    # 3. Build Adjacency List for Traversal
    children = {}
    for _, r in s.iterrows():
        h = r[head_col]
        i = r[index_col]
        if h not in children: children[h] = []
        children[h].append(i)

    # 4. Find Transitive Closure (All descendants of head_index)
    # Check if head exists
    if start_node not in s[index_col].values:
        print(f"Error: Head index {start_node} not found in sentence {sent_id}.")
        return

    subtree_indices = {start_node}
    queue = [start_node]

    while queue:
        curr = queue.pop(0)
        if curr in children:
            for child in children[curr]:
                if child not in subtree_indices:
                    subtree_indices.add(child)
                    queue.append(child)

    # 5. Filter and Sort by Linear Index
    clause_df = s[s[index_col].isin(subtree_indices)].copy()
    clause_df.sort_values(by=index_col, inplace=True)

    # 6. Print Output
    # Reconstruct readable text of just the clause
    tokens = clause_df[form_col].astype(str).tolist()
    print(f"Clause Text: \"{' '.join(tokens)}\"")
    print("-" * 40)

    # Print detailed tokens
    for _, r in clause_df.iterrows():
        i = r[index_col]
        form = r[form_col]
        upos = r.get(upos_col, "_")
        dep = r.get(deprel_col, "_")
        head = r.get(head_col, "_")
        
        # Mark the head visually
        marker = " (HEAD)" if i == start_node else ""
        print(f"{i}: {form}/{upos}/{head} [{dep}]{marker}")


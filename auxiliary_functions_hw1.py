from dataclasses import dataclass, asdict
from typing import Optional
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
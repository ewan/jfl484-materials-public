from dataclasses import dataclass, asdict
from typing import Optional, List
import traceback
import re

def get_text_from_conllu(filename):
    text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# text ='):
                text.append(line[8:].strip())
    return ' '.join(text)

def extract_words_from_conllu(path):
    """
    Read a CoNLL-U formatted file and extract the FORM (2nd field) from
    word annotation lines.

    A line is considered a word annotation if:
    - It is non-empty and not a comment (doesn't start with '#')
    - Splitting on tabs yields either 8 or 10 fields (CoNLL-U uses 10 fields, some variants use 8)
    - The first field can be converted to an integer (this excludes multiword tokens like "1-2" and empty nodes like "1.1")

    Returns:
        words (list of str): the extracted FORM values, in file order.
    """
    words = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue  # skip blank lines
            if line.startswith("#"):
                continue  # skip comment lines
            parts = line.split("\t")
            if len(parts) not in (8, 10):
                continue  # not a standard word-annotation line
            # Check first field is an integer (true word lines)
            try:
                _ = int(parts[0])
            except ValueError:
                # skip multiword tokens like "1-2" or empty-node IDs like "1.1"
                continue
            # Append the FORM (second field, index 1)
            words.append(parts[1])
    return words

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

def isClauseHead(tokens, currentToken, clauseHeads):
    """
    Find if a token is a clause head.

    Args:
        tokens: list of Token dataclass instances
        currentToken: index of the token whose status we are checking
        clauseHeads: list of dependencies that characterize clause heads
    """
    if tokens[currentToken].deprel in clauseHeads:
        return True
    elif tokens[currentToken].deprel in {'conj', 'parataxis'}:
        sentence = [t for t in tokens if t.sent_id == tokens[currentToken].sent_id]
        headIndex = tokens[currentToken].head # this is the "head" value for the current token 
        newToken = int(headIndex) - 1 # this is the corresponding index of the head in the list sentence
        return isClauseHead(sentence, newToken, clauseHeads)
    else:
        return False
    
def createObservation(tokens, currentToken):
    clauseHeads = {'root', 'advcl', 'acl', 'csubj', 'ccomp', 'xcomp'}
    if isClauseHead(tokens, currentToken, clauseHeads):
        newObservation = {
            'sent_id' : tokens[currentToken].sent_id,
            'title': tokens[currentToken].title,
            'text': tokens[currentToken].text,
            'translation': tokens[currentToken].translation,
            'form': tokens[currentToken].form,
            'deprel': tokens[currentToken].deprel
        }
        return newObservation
    else:
        return None

def process(fileList, createObservation):
    observations = [] # List of observations
    for f in fileList:
        try:
            with open(f, 'r', encoding='utf-8') as currentFile:
                tokens = buildTokenList(currentFile)
            for currentToken in range(len(tokens)):
                newObservation = createObservation(tokens, currentToken)
                if newObservation: observations.append(newObservation)
            print("Processed file: %s" % f)
        except Exception as e:
            print(e)
            traceback.print_exc()

    return observations
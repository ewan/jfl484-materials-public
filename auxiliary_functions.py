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

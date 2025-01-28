import re

try:
    import deepsmiles

    _has_deepsmiles = True
except ImportError as err:
    _has_deepsmiles = False
    DEEPSMILES_ERR = err
try:
    import selfies

    _has_selfies = True
except ImportError as err:
    _has_selfies = False
    SELFIES_ERR = err
try:
    import smizip

    _has_smizip = True
except ImportError as err:
    _has_smizip = False
    SMIZIP_ERR = err
try:
    import atomInSmiles as AIS

    _has_AIS = True
except ImportError as err:
    _has_AIS = False
    AIS_ERR = err
try:
    import safe

    _has_SAFE = True
except ImportError as err:
    _has_SAFE = False
    SAFE_ERR = err


class SMILESTokenizerChEMBL:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SMILES"

    def __init__(self, start_token="GO", end_token="EOS"):
        self.REGEXPS = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "2_ring_nums": re.compile(r"(%\d{2})"),
            "brcl": re.compile(r"(Br|Cl)"),
        }
        self.REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string."""

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, **kwargs):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                break
            if token != self.start_token:
                smi += token
        return smi


class SMILESTokenizerGuacaMol:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SMILES"

    def __init__(self, start_token="GO", end_token="EOS"):
        self.REGEXPS = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "brcl": re.compile(r"(Br|Cl)"),
        }
        self.REGEXP_ORDER = ["brackets", "brcl"]
        self.start_token = start_token
        self.end_token = end_token
        self.encode_dict = {
            "Br": "Y",
            "Cl": "X",
            "Si": "A",
            "Se": "Z",
            "@@": "R",
            "se": "E",
        }
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string."""
        for symbol, token in self.encode_dict.items():
            data = data.replace(symbol, token)

        tokens = list(data)
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, **kwargs):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                break
            if token != self.start_token:
                smi += token

        for symbol, token in self.decode_dict.items():
            smi = smi.replace(symbol, token)
        return smi


class SMILESTokenizerEnamine:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMER = "SMILES"

    def __init__(self, start_token="GO", end_token="EOS"):
        smiles_atoms = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ca",
            "Mn",
            "Fe",
            "Zn",
            "Se",
            "Br",
            "Pd",
            "Ag",
            "Cd",
            "I",
            "Hg",
            "b",
            "c",
            "n",
            "o",
            "p",
            "s",
        ]
        special_tokens = [
            "<pad>",
            "$",
            "^",
            "#",
            "%",
            "+",
            "[",
            "]",
            "(",
            ")",
            "-",
            ".",
            "@",
            "*",
            ":",
            ">",
            "/",
            "\\",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "=",
        ]
        sorted_atoms = sorted(smiles_atoms, key=len, reverse=True)
        special_tokens = [re.escape(token) for token in special_tokens]
        regex_pattern = "|".join(special_tokens + sorted_atoms)
        regex_pattern = r"(" + regex_pattern + ")"
        smiles_regex = re.compile(regex_pattern)
        self.REGEXP = re.compile(smiles_regex)
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string."""

        def split_by(data, regexp):
            if not regexp:
                return list(data)
            splitted = regexp.split(data)

            tokens = []
            for split in splitted:
                tokens.append(split)
            return list(filter(None, regexp.split(data)))

        tokens = split_by(data, self.REGEXP)
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                break
            if token != self.start_token:
                smi += token
        return smi


class DeepSMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "deepSMILES"

    def __init__(
        self,
        start_token="GO",
        end_token="EOS",
        rings=True,
        branches=True,
        compress=False,
    ):
        if not _has_deepsmiles:
            raise RuntimeError(
                "DeepSMILES library not found, please install with pip install deepsmiles."
            ) from DEEPSMILES_ERR
        self.converter = deepsmiles.Converter(rings=rings, branches=branches)
        self.run_compression = compress
        self.REGEXPS = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "brcl": re.compile(r"(Br|Cl)"),
        }
        self.REGEXP_ORDER = ["brackets", "brcl"]
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string via conversion to deepSMILES."""
        data = self.converter.encode(data)
        if self.run_compression:
            data = self.compress(data)

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes a deepSMILES string followed by conversion to SMILES."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        if convert_to_smiles:
            try:
                if self.run_compression:
                    smi = self.decompress(smi)
                smi = self.converter.decode(smi)
            except Exception:  # deepsmiles.DecodeError doesn't capture IndexError?
                smi = None
        return smi

    def compress(self, dsmi):
        """Compresses a deepSMILES string.

        > compress("C)C")
        'C)1C'
        > compress("C)))C")
        'C)3C'
        > compress("C))))))))))C")
        'C)10C'
        """
        compressed = []
        N = len(dsmi)
        i = 0
        while i < N:
            x = dsmi[i]
            compressed.append(x)
            if x == ")":
                start = i
                while i + 1 < N and dsmi[i + 1] == ")":
                    i += 1
                compressed.append(str(i + 1 - start))
            i += 1
        return "".join(compressed)

    def decompress(self, cdsmi):
        """Decompresses a compressed deepSMILES string.

        > decompress("C)1C")
        'C)C'
        > decompress("C)3C")
        'C)))C'
        > decompress("C)10C")
        'C))))))))))C'
        > decompress("C)C")
        Traceback (most recent call last):
            ...
        ValueError: A number should follow the parenthesis in C)C
        > decompress("C)")
        Traceback (most recent call last):
            ...
        ValueError: A number should follow the parenthesis in C)
        """
        decompressed = []
        N = len(cdsmi)
        i = 0
        while i < N:
            x = cdsmi[i]
            if x == ")":
                start = i
                while i + 1 < N and cdsmi[i + 1].isdigit():
                    i += 1
                if i == start:
                    raise ValueError(
                        f"A number should follow the parenthesis in {cdsmi}"
                    )
                number = int(cdsmi[start + 1 : i + 1])
                decompressed.append(")" * number)
            else:
                decompressed.append(x)
            i += 1
        return "".join(decompressed)


class SELFIESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SELFIES"

    def __init__(self, start_token="GO", end_token="EOS"):
        if not _has_selfies:
            raise RuntimeError(
                "SELFIES library not found, please install with pip install selfies ."
            ) from SELFIES_ERR
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string via conversion to SELFIES."""
        data = selfies.encoder(data)
        tokens = list(selfies.split_selfies(data))
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes a SELFIES string followed by conversion to SMILES."""
        smi = ""
        for token in tokens:
            if token == self.start_token:
                break
            if token != self.end_token:
                smi += token
        if convert_to_smiles:
            try:
                smi = selfies.decoder(smi)
            except Exception:
                smi = None
        return smi


class AISTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "AIS"

    def __init__(self, start_token="GO", end_token="EOS"):
        if not _has_AIS:
            raise RuntimeError(
                "atomInSmiles library not found, please install with pip install atomInSmiles."
            ) from AIS_ERR
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string via conversion to atomInSmiles."""
        data = AIS.encode(data)
        tokens = data.split(" ")
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes an atomInSmiles string followed by conversion to SMILES."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                smi = smi.rstrip()
                break
            if token != self.start_token:
                smi += token + " "
        if convert_to_smiles:
            try:
                smi = AIS.decode(smi)
            except Exception:
                smi = ""
        return smi


class SAFETokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SAFE"

    def __init__(self, start_token="GO", end_token="EOS"):
        if not _has_SAFE:
            raise RuntimeError(
                "SAFE library not found, please install with pip install safe-mol."
            ) from SAFE_ERR
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string via conversion to atomInSmiles."""
        data = safe.encode(data)
        tokens = safe.split(data)
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Untokenizes an atomInSmiles string followed by conversion to SMILES."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                break
            if token != self.start_token:
                smi += token
        if convert_to_smiles:
            try:
                smi = safe.decode(smi)
            except Exception:
                smi = None
        return smi


class SmiZipTokenizer:
    """Deals with the tokenization and untokenization of SmiZipped SMILES."""

    GRAMMAR = "SmiZip"

    def __init__(self, ngrams, start_token="GO", end_token="EOS"):
        if not _has_smizip:
            raise RuntimeError(
                "smizip library not found, please install with pip install smizip."
            ) from SMIZIP_ERR
        self.zipper = smizip.SmiZip(ngrams)
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=False):
        """Tokenizes a SMILES string via conversion to SmiZip tokens."""
        tokens = self.zipper.zip(data, format=1)  # format=1 returns the tokens
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, convert_to_smiles=True):
        """Join the SmiZip tokens to create a SMILES string."""
        ntokens = []
        for token in tokens:
            if token == self.end_token:
                break
            if token == self.start_token:
                continue
            ntokens.append(token)
        if convert_to_smiles:
            smi = "".join(ntokens)
        else:
            smi = ",".join(ntokens)
        return smi


class AsciiSMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES.

    Uses ASCII characters as tokens.
    """

    def __init__(self, start_token="^", end_token="$"):
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        tokens = list(data)
        if with_begin_and_end:
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                break
            if token != self.start_token:
                smi += token
        return smi

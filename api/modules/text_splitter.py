from enum import Enum
from typing import List

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# The algorithm to use for splitting texts
class SplitterTypeEnum(Enum):
    CHARACTER = 'CharacterTextSplitter'
    RECURSIVE = 'RecursiveTextSplitter'

class TextSplitter:
    splitter: CharacterTextSplitter | RecursiveCharacterTextSplitter

    def __init__(self, splitterType: SplitterTypeEnum, chunk_size: int, chunk_overlap: float):
        if splitterType not in SplitterTypeEnum:
            raise ValueError('Invalid splitter type')

        if chunk_size <= 0:
            raise ValueError('Chunk size must be greater than 0')

        if chunk_overlap < 0.0 or chunk_overlap > 1.0:
            raise ValueError('Chunk overlap must be between 0.0 and 1.0')

        if splitterType == SplitterTypeEnum.CHARACTER:
            self.splitter = CharacterTextSplitter(separator = " ",
                                                  chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        elif splitterType == SplitterTypeEnum.RECURSIVE:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap)
        else:
            raise RuntimeError('Could not initialize text splitter')

    # Text can be some arbitrary text
    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

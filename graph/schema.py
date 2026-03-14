from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Concept:
    id: str
    title: str
    aliases: List[str]
    difficulty: str
    definitions: List[str]
    formulas: List[str] = None  


@dataclass
class Resource:
    id: str
    type: str          
    url: str
    title: str


@dataclass
class Example:
    id: str
    text: str
    source_url: str
    related_concepts: List[str]

"""Datasets: JSONL in, immutable content-hashed snapshot out.

Each line is a JSON object:
    {"id": "q1",                      # optional, defaults to the line number
     "input": "How long do refunds take?",   # required (aliases: prompt, question)
     "reference": "5 business days",  # optional (alias: answer)
     "reference_docs": ["..."],       # optional texts the answer should be grounded in
     "relevant_doc_ids": ["d1"],      # optional labels for retrieval metrics
     "metadata": {"topic": "billing"}}
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Example:
    id: str
    input: str
    reference: Optional[str] = None
    reference_docs: Tuple[str, ...] = ()
    relevant_doc_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "input": self.input, "reference": self.reference,
            "reference_docs": list(self.reference_docs),
            "relevant_doc_ids": list(self.relevant_doc_ids), "metadata": self.metadata,
        }


@dataclass
class Dataset:
    name: str
    examples: List[Example]

    @property
    def content_hash(self) -> str:
        blob = json.dumps([e.to_dict() for e in self.examples], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _first(row: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if row.get(k) is not None:
            return row[k]
    return None


def load_jsonl(path: str, name: Optional[str] = None) -> Dataset:
    examples: List[Example] = []
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: each line must be a JSON object")
            text = _first(row, "input", "prompt", "question")
            if not isinstance(text, str) or not text.strip():
                raise DatasetError(f"{path}:{lineno}: missing required field 'input'")
            ex_id = str(row.get("id", len(examples) + 1))
            if ex_id in seen:
                raise DatasetError(f"{path}:{lineno}: duplicate id {ex_id!r}")
            seen.add(ex_id)
            ref = _first(row, "reference", "answer")
            examples.append(Example(
                id=ex_id, input=text,
                reference=None if ref is None else str(ref),
                reference_docs=tuple(str(d) for d in row.get("reference_docs") or ()),
                relevant_doc_ids=tuple(str(d) for d in row.get("relevant_doc_ids") or ()),
                metadata=row.get("metadata") or {},
            ))
    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return Dataset(name=name or path.replace("\\", "/").rsplit("/", 1)[-1], examples=examples)


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text, "metadata": self.metadata}


@dataclass
class Corpus:
    name: str
    docs: List[Doc]

    @property
    def content_hash(self) -> str:
        blob = json.dumps([d.to_dict() for d in self.docs], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_corpus_jsonl(path: str, name: Optional[str] = None) -> Corpus:
    """Documents for retrieval, one JSON object per line: {"id", "text", optional "title", "metadata"}."""
    docs: List[Doc] = []
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
            if not isinstance(row, dict) or not str(row.get("id", "")).strip():
                raise DatasetError(f"{path}:{lineno}: each document needs an 'id'")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise DatasetError(f"{path}:{lineno}: missing required field 'text'")
            doc_id = str(row["id"])
            if doc_id in seen:
                raise DatasetError(f"{path}:{lineno}: duplicate document id {doc_id!r}")
            seen.add(doc_id)
            docs.append(Doc(id=doc_id, text=row["text"], title=row.get("title"),
                            metadata=row.get("metadata") or {}))
    if not docs:
        raise DatasetError(f"{path}: corpus is empty")
    return Corpus(name=name or path.replace("\\", "/").rsplit("/", 1)[-1], docs=docs)

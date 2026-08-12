import os
import re
from typing import List, Dict, Any, Tuple


SYNONYMS = {
    "os": ["operating system", "operating systems", "os"],
    "dbms": ["database", "databases", "dbms", "database management system"],
    "cn": ["computer networks", "networking", "cn"],
    "algo": ["algorithms", "algo"],
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    for key, syns in SYNONYMS.items():
        for syn in syns:
            text = re.sub(rf"\b{re.escape(syn)}\b", key, text)
    return text


class DocumentResolver:
    """Resolves explicit, partial, or contextual document references against vector store document metadata."""

    def get_catalog(self, vector_store) -> List[Dict[str, Any]]:
        if not vector_store:
            return []
        all_docs = vector_store.get_all_documents()
        catalog_map = {}

        for doc in all_docs:
            fid = doc.metadata.get("document_id") or doc.metadata.get("source_filename")
            if not fid or fid in catalog_map:
                continue

            source_filename = doc.metadata.get("source_filename", fid)
            document_type = doc.metadata.get("document_type", "book")
            name_no_ext = os.path.splitext(source_filename)[0]

            catalog_map[fid] = {
                "document_id": fid,
                "source_filename": source_filename,
                "document_type": document_type,
                "name_no_ext": name_no_ext,
                "normalized_filename": normalize_text(source_filename),
                "normalized_name": normalize_text(name_no_ext.replace("_", " ")),
            }
        return list(catalog_map.values())

    def _resolve_single_phrase(self, phrase: str, catalog: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        p_norm = normalize_text(phrase)
        if not p_norm or not catalog:
            return [], []

        # 1. Exact filename match
        for item in catalog:
            if item["source_filename"].lower() == p_norm or item["normalized_filename"] == p_norm or item["source_filename"].lower() in p_norm:
                return [item["document_id"]], []

        # 2. Exact name without extension match
        for item in catalog:
            if item["normalized_name"] == p_norm:
                return [item["document_id"]], []

        # 3. Partial filename / title / type scoring match
        scored_candidates = []
        for item in catalog:
            score = 0
            doc_name = item["normalized_name"]
            dtype = item["document_type"]

            # Type filtering cue
            if "exam" in p_norm or "paper" in p_norm:
                if dtype == "exam_paper":
                    score += 2
                else:
                    score -= 2
            elif "book" in p_norm or "textbook" in p_norm or "notes" in p_norm:
                if dtype == "book":
                    score += 1

            # Match significant words
            words = [w for w in re.split(r"[_\-\s\.]+", p_norm) if len(w) >= 2 and w not in ["the", "a", "an", "document", "file"]]
            matched_words = [w for w in words if w in doc_name]

            if matched_words:
                score += len(matched_words) * 3


            if score > 0:
                scored_candidates.append((score, item["document_id"]))

        if not scored_candidates:
            return [], []

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_score = scored_candidates[0][0]

        top_matches = [cid for s, cid in scored_candidates if s == top_score]
        if len(top_matches) == 1:
            return top_matches, []
        elif len(top_matches) > 1:
            return [], top_matches

        return [], []

    def resolve(
        self,
        query: str,
        catalog: List[Dict[str, Any]],
        intent: str = "question",
        chat_history: list = None,
    ) -> Tuple[List[str], List[str]]:
        if not catalog:
            return [], []

        q_norm = normalize_text(query)

        # 1. Multi-document comparison split
        if intent == "comparison" or "compare" in q_norm or " versus " in q_norm or " vs " in q_norm:
            if "exam paper" in q_norm or "exam papers" in q_norm or "papers" in q_norm:
                papers = [item["document_id"] for item in catalog if item["document_type"] == "exam_paper"]
                if len(papers) >= 2:
                    return papers[:2], []
            elif "books" in q_norm or "textbooks" in q_norm or "documents" in q_norm:
                books = [item["document_id"] for item in catalog if item["document_type"] == "book"]
                if len(books) >= 2:
                    return books[:2], []

            parts = re.split(r"\b(?:compare|with|and|versus|vs|to|between)\b", q_norm, flags=re.I)
            parts = [p.strip() for p in parts if p.strip() and p.strip() not in ["the", "a", "an", "difference", "differences", "these", "two"]]
            resolved_ids = []
            ambiguous_all = []

            for p in parts:
                r_ids, a_ids = self._resolve_single_phrase(p, catalog)
                if r_ids:
                    for rid in r_ids:
                        if rid not in resolved_ids:
                            resolved_ids.append(rid)
                elif a_ids:
                    ambiguous_all.extend(a_ids)

            if len(resolved_ids) >= 2:
                return resolved_ids, []
            elif resolved_ids:
                return resolved_ids, ambiguous_all


        # 2. Direct filename substring match on full catalog
        exact_matches = []
        for item in catalog:
            if item["source_filename"].lower() in q_norm or item["normalized_name"] in q_norm:
                exact_matches.append(item["document_id"])

        if len(exact_matches) == 1:
            return exact_matches, []
        elif len(exact_matches) > 1:
            exact_file = [item["document_id"] for item in catalog if item["source_filename"].lower() in q_norm]
            if len(exact_file) == 1:
                return exact_file, []
            return [], exact_matches

        # 3. Single phrase scoring resolution
        resolved, amb = self._resolve_single_phrase(q_norm, catalog)
        if resolved or amb:
            return resolved, amb

        # 4. Conversation history context fallback ("that book", "the notes", "previous document", "it")
        if chat_history and ("book" in q_norm or "notes" in q_norm or "document" in q_norm or "paper" in q_norm or "previous" in q_norm or "it" in q_norm):
            for msg in reversed(chat_history):
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if content:
                    hist_res, _ = self._resolve_single_phrase(content, catalog)
                    if hist_res:
                        return hist_res, []

        return [], []

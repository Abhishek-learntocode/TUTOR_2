import json
import re
from app.models.state import QueryAnalysis, Message


def is_conversational_query(query: str) -> bool:
    """Detect if a query is conversational (user name, greeting, chat history question, small talk)."""
    q_lower = query.lower().strip()
    conversational_patterns = [
        r"\b(my name|who am i|what is my name|remember my name|do you know my name|call me|i am \w+)\b",
        r"\b(hello|hi|hey|good morning|good afternoon|good evening|how are you|who are you|what can you do|thank you|thanks)\b",
        r"\b(what did i ask|what was my previous|repeat what i said|what was my first question)\b",
    ]
    for pattern in conversational_patterns:
        if re.search(pattern, q_lower):
            return True
    return False


def resolve_reference_fallback(query: str, chat_history: list) -> str:
    """Fallback reference resolution if LLM failed to resolve pronouns like 'its', 'it', 'this', 'that'."""
    pronoun_pattern = r"\b(its|it|this|that|these|those|their)\b"
    if not re.search(pronoun_pattern, query, re.I) or not chat_history:
        return query

    last_user_turn = ""
    for msg in reversed(chat_history):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if role == "user":
            if not is_conversational_query(content):
                last_user_turn = content
                break

    if not last_user_turn:
        return query

    clean = re.sub(
        r"^(what|how|why|where|when|who|is|are|can|could|explain|describe|tell me about)\s+(is|are|a|an|the)?\s*",
        "",
        last_user_turn,
        flags=re.I,
    )
    clean = clean.strip("? .!\n")

    if clean:
        resolved = re.sub(pronoun_pattern, f"the {clean}", query, count=1, flags=re.I)
        return resolved
    return query


def detect_intent(query: str) -> str:
    q_lower = query.lower()
    if any(k in q_lower for k in ["summarise", "summarize", "summary"]):
        return "summarization"
    elif any(k in q_lower for k in ["compare", "difference between", " versus ", " vs "]):
        return "comparison"
    elif any(k in q_lower for k in ["find the document", "find the exam paper", "find the book", "show me the notes"]):
        return "document_lookup"
    return "question"


def detect_document_references(query: str) -> tuple[bool, list[str]]:
    q_lower = query.lower()
    doc_kw_patterns = [
        r"\b[\w\-]+\.(?:txt|pdf|md|docx?)\b",
        r"\b(?:the|this|that|my|previous)?\s*(?:os|dbms|cn|operating system|database)?\s*(?:notes|book|textbook|exam paper|paper|document)\b",
        r"\b(?:book|notes|textbook|exam paper|paper|document)\b",
    ]
    refs = []
    for pat in doc_kw_patterns:
        matches = re.findall(pat, q_lower)
        if matches:
            for m in matches:
                if isinstance(m, str) and len(m.strip()) > 1:
                    refs.append(m.strip())

    requires_res = bool(refs or any(ext in q_lower for ext in [".txt", ".pdf", ".md"]) or any(kw in q_lower for kw in ["book", "notes", "paper", "document", "this document", "that document", "previous document"]))
    return requires_res, list(set(refs))


def map_to_operation(intent: str, requires_res: bool) -> str:
    if intent == "summarization":
        return "summarize"
    elif intent == "comparison":
        return "compare"
    elif requires_res:
        return "document_qa"
    return "normal_qa"


class QueryAnalyzer:
    """Analyzes queries using LLM + rule heuristics to classify as single_hop, multi_hop, or conversational."""

    def __init__(self, llm):
        self.llm = llm

    def analyze(self, query: str, chat_history: list[Message] = None) -> QueryAnalysis:
        if is_conversational_query(query):
            return QueryAnalysis(
                operation="conversational",
                intent="question",
                query_type="conversational",
                sub_queries=[query],
                document_references=[],
                requires_document_resolution=False,
            )

        intent = detect_intent(query)
        requires_res, doc_refs = detect_document_references(query)
        operation = map_to_operation(intent, requires_res)

        query_lower = query.lower()
        has_multi_hop_cue = (
            intent == "comparison"
            or "compare" in query_lower
            or "difference between" in query_lower
            or " versus " in query_lower
            or " vs " in query_lower
            or bool(re.search(r"chapter\s+\d+.*chapter\s+\d+", query, re.I))
        )

        history_text = ""
        if chat_history:
            formatted = []
            for msg in chat_history:
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if role and content:
                    role_label = "User" if role == "user" else "Assistant"
                    formatted.append(f"{role_label}: {content}")
            if formatted:
                history_text = "Chat History:\n" + "\n".join(formatted) + "\n\n"

        prompt = (
            "You are an expert Query Analyzer for a technical RAG system.\n"
            "Analyze the current user question in light of the chat history (if provided).\n\n"
            "TASKS:\n"
            "1. Reference Resolution: Rewrite follow-up questions containing pronouns or implicit references "
            "(e.g., 'its', 'it', 'this', 'that', 'the former', 'that document', 'previous document', 'how does it work', 'what are its advantages') "
            "into self-contained, standalone search queries using the chat history.\n"
            "2. Classification:\n"
            "   - 'single_hop': Question asks about a single concept, definition, or direct topic.\n"
            "   - 'multi_hop': Answer requires connecting independently retrieved pieces of evidence from separate topics/chapters. "
            "(CRITICAL RULE: Multi-hop means connecting independently retrieved pieces of evidence, NOT simply that the question is long or contains multiple clauses).\n\n"
            "RULES FOR SUB-QUERIES:\n"
            "- For single_hop: Output exactly 1 standalone search query (with references resolved).\n"
            "- For multi_hop: Output the minimum necessary (2-3) distinct, independently retrievable sub-queries.\n"
            "- Avoid duplicate sub-queries or unnecessary decomposition.\n\n"
            "EXAMPLES:\n"
            "Chat History:\nUser: What is virtual memory?\nAssistant: Virtual memory is...\n"
            'User Question: "What are its advantages?"\n'
            '{"query_type": "single_hop", "sub_queries": ["What are the advantages of virtual memory?"]}\n\n'
            "Chat History:\nUser: What is page fault?\nAssistant: A page fault occurs...\n"
            'User Question: "Compare it with thrashing."\n'
            '{"query_type": "multi_hop", "sub_queries": ["What is a page fault in operating systems?", "What is thrashing in operating systems?"]}\n\n'
            f"{history_text}"
            f'User Question: "{query}"\n\n'
            "Respond ONLY with a valid JSON object:"
        )


        try:
            raw_response = self.llm.generate(prompt, context=[])
            cleaned = re.sub(r"```(?:json)?", "", raw_response).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                q_type = data.get("query_type", "single_hop")
                sub_q = data.get("sub_queries", [])

                if has_multi_hop_cue and q_type == "single_hop":
                    q_type = "multi_hop"

                if q_type == "multi_hop" and (not isinstance(sub_q, list) or len(sub_q) < 2):
                    sub_q = [
                        f"Information regarding the first aspect of: {query}",
                        f"Information regarding the second aspect of: {query}",
                    ]
                elif q_type == "single_hop":
                    if not isinstance(sub_q, list) or not sub_q or not isinstance(sub_q[0], str):
                        single_res = resolve_reference_fallback(query, chat_history)
                        sub_q = [single_res]
                    else:
                        single_res = resolve_reference_fallback(sub_q[0], chat_history)
                        sub_q = [single_res]

                return QueryAnalysis(
                    operation=operation,
                    intent=intent,
                    query_type=q_type,
                    sub_queries=sub_q,
                    document_references=doc_refs,
                    requires_document_resolution=requires_res,
                )
        except Exception as e:
            print(f"[QueryAnalyzer Warning] Analysis failed, defaulting to single_hop: {e}")

        fallback_q = resolve_reference_fallback(query, chat_history)
        q_type = "multi_hop" if has_multi_hop_cue else "single_hop"
        sub_q = [
            f"Information regarding first topic in: {fallback_q}",
            f"Information regarding second topic in: {fallback_q}",
        ] if has_multi_hop_cue else [fallback_q]

        return QueryAnalysis(
            operation=operation,
            intent=intent,
            query_type=q_type,
            sub_queries=sub_q,
            document_references=doc_refs,
            requires_document_resolution=requires_res,
        )





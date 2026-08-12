import json
import re

from app.models.state import QueryAnalysis, Message


class QueryAnalyzer:
    """
    Analyzes user queries using lightweight rule heuristics
    and the local LLM when necessary.

    Responsibilities:
        1. Conversational/reference resolution
        2. Intent detection
        3. Operation detection
        4. Single-hop / multi-hop classification
        5. Sub-query generation
        6. Document-reference detection
    """

    def __init__(self, llm):
        self.llm = llm

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def analyze(
        self,
        query: str,
        chat_history: list[Message] = None,
    ) -> QueryAnalysis:
        """
        Analyze the user query.

        Conversation references are resolved first.

        Example:

            Previous:
                What is virtual memory?

            Current:
                Why is it useful?

            Internal query:
                Why is virtual memory useful?
        """

        original_query = query.strip()

        # ------------------------------------------------------
        # 1. Resolve conversational references
        # ------------------------------------------------------

        resolved_query = self.resolve_conversation_reference(
            original_query,
            chat_history,
        )

        query = resolved_query
        query_lower = query.lower().strip()

        # ------------------------------------------------------
        # 2. Conversational / non-RAG detection
        # ------------------------------------------------------

        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "how are you",
            "thanks",
            "thank you",
        ]

        is_greeting = (
            query_lower in greetings
            or any(
                query_lower.startswith(
                    greeting + " "
                )
                for greeting in ["hi", "hello", "hey"]
            )
        )

        is_personal_intro = any(
            phrase in query_lower
            for phrase in [
                "my name is",
                "i am ",
                "call me ",
                "i study",
                "i work",
            ]
        )

        is_personal_question = any(
            phrase in query_lower
            for phrase in [
                "what is my name",
                "who am i",
                "what do i study",
                "what did i say",
                "my name?",
            ]
        )

        if (
            is_greeting
            or is_personal_intro
            or is_personal_question
        ):
            return QueryAnalysis(
                operation="conversational",
                intent="question",
                query_type="conversational",
                sub_queries=[query],
                requires_document_resolution=False,
            )

        # ------------------------------------------------------
        # 3. Basic intent / operation cues
        # ------------------------------------------------------

        is_summarize = (
            "summarise" in query_lower
            or "summarize" in query_lower
            or "summary" in query_lower
        )

        is_compare = (
            "compare" in query_lower
            or "versus" in query_lower
            or " vs " in query_lower
        )

        doc_keywords = [
            "note",
            "notes",
            "book",
            "textbook",
            "exam",
            "paper",
            "document",
            "doc",
            "pdf",
        ]

        # Do NOT treat every "." as a document reference.
        # This prevents normal questions ending with "." from
        # accidentally entering document_qa.
        has_filename_pattern = bool(
            re.search(
                r"\b[\w\-. ]+\.(?:txt|pdf|md|docx?|pptx?|xlsx?)\b",
                query,
                re.IGNORECASE,
            )
        )

        has_doc_keyword = any(
            keyword in query_lower
            for keyword in doc_keywords
        )

        has_doc_ref = (
            has_filename_pattern
            or has_doc_keyword
        )

        # ------------------------------------------------------
        # 4. Multi-hop detection
        # ------------------------------------------------------

        chapter_multi_hop = bool(
            re.search(
                r"chapter\s+\d+.*chapter\s+\d+",
                query,
                re.IGNORECASE | re.DOTALL,
            )
        )

        difference_cue = (
            "difference between" in query_lower
        )

        compare_cue = (
            "compare" in query_lower
            or "versus" in query_lower
            or " vs " in query_lower
        )

        relationship_cue = bool(
            re.search(
                r"\bhow does .+?"
                r"\b(?:relate to|affect|influence|"
                r"connect to)\b",
                query_lower,
                re.IGNORECASE,
            )
        )

        multi_question_cue = bool(
            re.search(
                r",\s*(?:and\s+)?"
                r"(?:what|why|how|where|when|which)\b",
                query,
                re.IGNORECASE | re.DOTALL,
            )
        )

        has_multi_hop_cue = (
            chapter_multi_hop
            or difference_cue
            or compare_cue
            or relationship_cue
            or multi_question_cue
        )

        # ------------------------------------------------------
        # 5. Explicit multi-hop
        # ------------------------------------------------------

        if (
            has_multi_hop_cue
            and not is_summarize
        ):
            sub_queries = self._build_multi_hop_queries(
                query
            )

            document_references = (
                self._extract_document_references(
                    query,
                    doc_keywords,
                )
            )

            return QueryAnalysis(
                operation=(
                    "compare"
                    if is_compare
                    else "normal_qa"
                ),
                intent=(
                    "comparison"
                    if is_compare
                    else "question"
                ),
                query_type="multi_hop",
                sub_queries=sub_queries,
                document_references=(
                    document_references
                ),
                requires_document_resolution=bool(
                    document_references
                ),
            )

        # ------------------------------------------------------
        # 6. Summarization
        # ------------------------------------------------------

        if is_summarize:
            document_references = (
                self._extract_document_references(
                    query,
                    doc_keywords,
                )
            )

            return QueryAnalysis(
                operation="summarize",
                intent="summarization",
                query_type="single_hop",
                sub_queries=[query],
                document_references=(
                    document_references
                    if document_references
                    else [query]
                ),
                requires_document_resolution=True,
            )

        # ------------------------------------------------------
        # 7. Comparison
        # ------------------------------------------------------

        if is_compare:
            sub_queries = (
                self._build_multi_hop_queries(
                    query
                )
            )

            return QueryAnalysis(
                operation="compare",
                intent="comparison",
                query_type="multi_hop",
                sub_queries=sub_queries,
                document_references=(
                    self._extract_document_references(
                        query,
                        doc_keywords,
                    )
                ),
                requires_document_resolution=(
                    has_doc_ref
                ),
            )

        # ------------------------------------------------------
        # 8. Explicit document query
        # ------------------------------------------------------

        if has_doc_ref:
            document_references = (
                self._extract_document_references(
                    query,
                    doc_keywords,
                )
            )

            return QueryAnalysis(
                operation="document_qa",
                intent="document_lookup",
                query_type="single_hop",
                sub_queries=[query],
                document_references=(
                    document_references
                    if document_references
                    else [query]
                ),
                requires_document_resolution=True,
            )

        # ------------------------------------------------------
        # 9. LLM-based analysis fallback
        # ------------------------------------------------------

        history_context = ""

        if chat_history:
            recent_history = chat_history[-4:]

            history_text = "\n".join(
                f"{message.role}: {message.content}"
                for message in recent_history
            )

            history_context = (
                "\nRecent Chat History:\n"
                f"{history_text}\n"
            )

        prompt = (
            "You are a Query Analyzer for an "
            "Intelligent RAG system.\n\n"

            "Classify the user input into JSON with:\n"
            "- operation\n"
            "- intent\n"
            "- query_type\n"
            "- sub_queries\n"
            "- document_references\n\n"

            "OPERATIONS:\n"
            "- conversational: greetings, introductions, "
            "or questions about chat history/user identity.\n"
            "- summarize: asking to summarize a document.\n"
            "- compare: comparing two concepts or documents.\n"
            "- document_qa: asking about a named document.\n"
            "- normal_qa: standard topic/concept questions.\n\n"

            "QUERY TYPES:\n"
            "- single_hop: one topic or direct question.\n"
            "- multi_hop: combines multiple distinct "
            "topics or requires multiple pieces of evidence.\n"
            "- conversational: non-RAG conversation.\n\n"

            "IMPORTANT:\n"
            "Do not create fake sub-queries such as "
            "'First aspect of...' or 'Second aspect of...'.\n"
            "If the query is single-hop, return the original "
            "query as the only sub-query.\n\n"

            f"{history_context}"

            f'User Input: "{query}"\n\n'

            "Respond ONLY with valid JSON:\n"

            '{'
            '"operation": "normal_qa", '
            '"intent": "question", '
            '"query_type": "single_hop", '
            '"sub_queries": ["..."], '
            '"document_references": []'
            '}'
        )

        try:
            raw_response = self.llm.generate(
                prompt,
                context=[],
            )

            cleaned = re.sub(
                r"```(?:json)?",
                "",
                raw_response,
                flags=re.IGNORECASE,
            ).strip("` \n")

            json_match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            if json_match:
                data = json.loads(
                    json_match.group(0)
                )

                operation = data.get(
                    "operation",
                    "normal_qa",
                )

                intent = data.get(
                    "intent",
                    "question",
                )

                query_type = data.get(
                    "query_type",
                    "single_hop",
                )

                sub_queries = data.get(
                    "sub_queries",
                    [query],
                )

                document_references = data.get(
                    "document_references",
                    [],
                )

                # --------------------------------------------------
                # Validate query type
                # --------------------------------------------------

                if query_type not in {
                    "single_hop",
                    "multi_hop",
                    "conversational",
                }:
                    query_type = "single_hop"

                # --------------------------------------------------
                # Validate sub-queries
                # --------------------------------------------------

                if not isinstance(
                    sub_queries,
                    list,
                ):
                    sub_queries = [query]

                sub_queries = [
                    str(sub_query).strip()
                    for sub_query in sub_queries
                    if str(sub_query).strip()
                ]

                # Never allow empty sub-query list.
                if not sub_queries:
                    sub_queries = [query]

                # --------------------------------------------------
                # Multi-hop validation
                # --------------------------------------------------

                if query_type == "multi_hop":

                    if len(sub_queries) < 2:
                        sub_queries = (
                            self._build_multi_hop_queries(
                                query
                            )
                        )

                    # If the LLM returned meaningless
                    # fallback phrases, rebuild them.
                    if any(
                        self._is_fake_subquery(
                            sub_query
                        )
                        for sub_query in sub_queries
                    ):
                        sub_queries = (
                            self._build_multi_hop_queries(
                                query
                            )
                        )

                elif query_type == "single_hop":
                    sub_queries = [query]

                elif query_type == "conversational":
                    sub_queries = [query]

                # --------------------------------------------------
                # Clean and deduplicate
                # --------------------------------------------------

                cleaned_sub_queries = []

                seen_sub_queries = set()

                for sub_query in sub_queries:

                    cleaned_sub_query = (
                        self._clean_sub_query(
                            sub_query
                        )
                    )

                    normalized = (
                        cleaned_sub_query.lower()
                    )

                    if normalized in seen_sub_queries:
                        continue

                    seen_sub_queries.add(
                        normalized
                    )

                    cleaned_sub_queries.append(
                        cleaned_sub_query
                    )

                sub_queries = (
                    cleaned_sub_queries
                    if cleaned_sub_queries
                    else [query]
                )

                # --------------------------------------------------
                # Validate document references
                # --------------------------------------------------

                if not isinstance(
                    document_references,
                    list,
                ):
                    document_references = []

                document_references = [
                    str(reference).strip()
                    for reference in document_references
                    if str(reference).strip()
                ]

                requires_document_resolution = (
                    operation
                    in {
                        "summarize",
                        "compare",
                        "document_qa",
                    }
                    or bool(document_references)
                )

                return QueryAnalysis(
                    operation=operation,
                    intent=intent,
                    query_type=query_type,
                    sub_queries=sub_queries,
                    document_references=(
                        document_references
                    ),
                    requires_document_resolution=(
                        requires_document_resolution
                    ),
                )

        except Exception as exc:
            print(
                "[QueryAnalyzer Warning] "
                f"Analysis failed, defaulting: {exc}"
            )

        # ------------------------------------------------------
        # Safe final fallback
        # ------------------------------------------------------

        return QueryAnalysis(
            operation="normal_qa",
            intent="question",
            query_type="single_hop",
            sub_queries=[query],
            requires_document_resolution=False,
        )

    # ==========================================================
    # CONVERSATION REFERENCE RESOLUTION
    # ==========================================================

    def resolve_conversation_reference(
        self,
        query: str,
        chat_history: list[Message] | None = None,
    ) -> str:
        """
        Resolve conversational references using the local LLM.

        Example:

            User:
                What is virtual memory?

            User:
                Why is it useful?

            Resolved:
                Why is virtual memory useful?

        The resolver is intentionally triggered only when
        explicit conversational references are present.
        Normal standalone questions do not pay for another
        LLM call.
        """

        query = query.strip()

        if not query:
            return query

        if not chat_history:
            return query

        query_lower = query.lower()

        # ------------------------------------------------------
        # Explicit conversational references
        # ------------------------------------------------------

        reference_patterns = [
            r"\bit\b",
            r"\bits\b",
            r"\bthis\b",
            r"\bthat\b",
            r"\bthese\b",
            r"\bthose\b",
            r"\bthey\b",
            r"\bthem\b",
            r"\bthe above\b",
            r"\bthe previous\b",
            r"\bthe same\b",
            r"\bthe concept\b",
            r"\bthe topic\b",
            r"\bthe method\b",
            r"\bthe technique\b",
            r"\bthe approach\b",
            r"\bthe first one\b",
            r"\bthe second one\b",
            r"\bthe other one\b",
        ]

        has_reference = any(
            re.search(
                pattern,
                query_lower,
                re.IGNORECASE,
            )
            for pattern in reference_patterns
        )

        # ------------------------------------------------------
        # IMPORTANT:
        # Do not treat every short question as a follow-up.
        #
        # Examples such as:
        #   "What is virtual memory?"
        #   "What is paging?"
        #
        # are normal standalone queries and must NOT trigger
        # another LLM call.
        #
        # Only explicit conversational references trigger
        # the resolver.
        # ------------------------------------------------------

        if not has_reference:
            return query

        # ------------------------------------------------------
        # Recent conversation only
        # ------------------------------------------------------

        recent_history = chat_history[-6:]

        history_text = "\n".join(
            (
                f"{message.role}: "
                f"{message.content}"
            )
            for message in recent_history
        )

        prompt = f"""
You are a conversational query resolver for a RAG system.

Your ONLY task is to rewrite the CURRENT USER QUERY
into a standalone query using the recent conversation.

RULES:

1. Resolve references such as:
   - it
   - this
   - that
   - they
   - them
   - the above
   - the previous one
   - the concept
   - the topic
   - the method
   - the technique

2. Preserve the user's original intent.

3. Do NOT answer the question.

4. Do NOT explain your reasoning.

5. Do NOT invent information.

6. If the query is already standalone, return it
   unchanged.

7. Return ONLY the rewritten query.

8. Do not use quotation marks.

9. Keep the rewritten query concise.

RECENT CONVERSATION:
{history_text}

CURRENT USER QUERY:
{query}

STANDALONE QUERY:
""".strip()

        try:
            rewritten = self.llm.generate(
                prompt,
                context=[],
            )

            rewritten = (
                rewritten
                .strip()
                .strip('"')
                .strip("'")
            )

            rewritten = re.sub(
                r"\s+",
                " ",
                rewritten,
            ).strip()

            if not rewritten:
                return query

            # --------------------------------------------------
            # Reject answer-like / failure-like LLM responses.
            #
            # The local model can sometimes ignore the rewrite
            # instruction and return a normal RAG answer such as:
            #
            #   "I cannot find the answer in the provided context."
            #
            # That text must NEVER become the retrieval query.
            # --------------------------------------------------

            invalid_phrases = (
                "i cannot find the answer",
                "i can't find the answer",
                "cannot find the answer",
                "can't find the answer",
                "i don't know",
                "i do not know",
                "i cannot answer",
                "i can't answer",
                "there is no information",
                "no information is provided",
                "no relevant information",
                "insufficient information",
                "based on the context",
                "based on the provided context",
                "based on the given context",
                "the answer is",
                "here is the answer",
                "the rewritten query is",
                "rewritten query:",
                "here is the rewritten query",
                "the standalone query is",
                "standalone query:",
            )

            rewritten_lower = rewritten.lower()

            if any(
                phrase in rewritten_lower
                for phrase in invalid_phrases
            ):
                print(
                    "[Conversation Resolver] "
                    "Invalid LLM output; keeping original query."
                )
                return query

            # --------------------------------------------------
            # Reject obvious meta / refusal responses.
            # --------------------------------------------------

            invalid_prefix_patterns = (
                r"^i am unable",
                r"^i cannot",
                r"^i can't",
                r"^as an ai",
                r"^the provided context",
                r"^there is insufficient",
                r"^i don't have enough",
            )

            if any(
                re.search(
                    pattern,
                    rewritten_lower,
                )
                for pattern in invalid_prefix_patterns
            ):
                print(
                    "[Conversation Resolver] "
                    "Meta/refusal output; keeping original query."
                )
                return query

            # --------------------------------------------------
            # Protect against an excessively long response.
            # A rewrite should be close to query length, not a
            # paragraph of generated content.
            # --------------------------------------------------

            if len(rewritten) > max(
                len(query) * 4,
                500,
            ):
                print(
                    "[Conversation Resolver] "
                    "Output too long; keeping original query."
                )
                return query

            # --------------------------------------------------
            # Reject multi-sentence answer-like output.
            # This is a lightweight safety check, not a hard
            # requirement that every query end with '?'.
            # --------------------------------------------------

            sentence_count = len(
                re.findall(
                    r"[.!?]+(?:\s|$)",
                    rewritten,
                )
            )

            if (
                sentence_count >= 3
                and len(rewritten.split()) > 20
            ):
                print(
                    "[Conversation Resolver] "
                    "Output looks like an answer; "
                    "keeping original query."
                )
                return query

            print(
                "[CONVERSATION RESOLUTION]"
            )

            print(
                f"Original query: {query}"
            )

            print(
                f"Resolved query: {rewritten}"
            )

            return rewritten

        except Exception as exc:
            print(
                "[Conversation Resolver Warning] "
                f"Resolution failed: {exc}"
            )

            return query

    # ==========================================================
    # MULTI-HOP QUERY GENERATION
    # ==========================================================

    @staticmethod
    def _build_multi_hop_queries(
        query: str,
    ) -> list[str]:
        """
        Create focused sub-queries for obvious multi-hop
        questions.

        Example:

            How does paging enable virtual memory,
            and what limitation of physical memory does
            this address?

        Returns:

            [
                "How does paging enable virtual memory?",
                "What limitation of physical memory does this address?"
            ]
        """

        # ------------------------------------------------------
        # "X and what/why/how Y"
        # ------------------------------------------------------

        match = re.match(
            r"(.+?)(?:,\s*|\s+)and\s+"
            r"(what|why|how|where|when|which)\s+"
            r"(.+?)(?:\?)?$",
            query.strip(),
            re.IGNORECASE | re.DOTALL,
        )

        if match:

            first_part = (
                match.group(1)
                .strip()
            )

            second_word = (
                match.group(2)
                .strip()
            )

            second_part = (
                match.group(3)
                .strip()
            )

            first_query = (
                QueryAnalyzer._clean_sub_query(
                    first_part
                )
            )

            second_query = (
                QueryAnalyzer._clean_sub_query(
                    f"{second_word.capitalize()} "
                    f"{second_part}"
                )
            )

            return [
                first_query,
                second_query,
            ]

        # ------------------------------------------------------
        # Relationship questions
        # ------------------------------------------------------

        relationship_match = re.match(
            r"how does (.+?)\s+"
            r"(relate to|affect|influence|connect to)\s+"
            r"(.+?)(?:\?)?$",
            query.strip(),
            re.IGNORECASE | re.DOTALL,
        )

        if relationship_match:

            concept_a = (
                relationship_match
                .group(1)
                .strip()
            )

            concept_b = (
                relationship_match
                .group(3)
                .strip()
            )

            return [
                QueryAnalyzer._clean_sub_query(
                    f"What is {concept_a}"
                ),
                QueryAnalyzer._clean_sub_query(
                    f"What is {concept_b}"
                ),
            ]

        # ------------------------------------------------------
        # Difference / comparison
        # ------------------------------------------------------

        difference_match = re.search(
            r"difference between\s+(.+?)\s+"
            r"and\s+(.+?)(?:\?|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )

        if difference_match:

            concept_a = (
                difference_match
                .group(1)
                .strip()
            )

            concept_b = (
                difference_match
                .group(2)
                .strip()
            )

            return [
                QueryAnalyzer._clean_sub_query(
                    f"What is {concept_a}"
                ),
                QueryAnalyzer._clean_sub_query(
                    f"What is {concept_b}"
                ),
            ]

        # ------------------------------------------------------
        # Explicit comparison
        # ------------------------------------------------------

        comparison_match = re.search(
            r"compare\s+(.+?)\s+"
            r"(?:and|with|vs\.?|versus)\s+"
            r"(.+?)(?:\?|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )

        if comparison_match:

            concept_a = (
                comparison_match
                .group(1)
                .strip()
            )

            concept_b = (
                comparison_match
                .group(2)
                .strip()
            )

            return [
                QueryAnalyzer._clean_sub_query(
                    f"What is {concept_a}"
                ),
                QueryAnalyzer._clean_sub_query(
                    f"What is {concept_b}"
                ),
            ]

        # ------------------------------------------------------
        # Chapter queries
        # ------------------------------------------------------

        chapters = re.findall(
            r"chapter\s+(\d+)",
            query,
            re.IGNORECASE,
        )

        if len(chapters) >= 2:

            return [
                QueryAnalyzer._clean_sub_query(
                    f"What is discussed in "
                    f"Chapter {chapters[0]}"
                ),
                QueryAnalyzer._clean_sub_query(
                    f"What is discussed in "
                    f"Chapter {chapters[1]}"
                ),
            ]

        # ------------------------------------------------------
        # Multiple explicit questions
        # ------------------------------------------------------

        question_parts = re.split(
            r"\?\s+",
            query,
        )

        question_parts = [
            part.strip()
            for part in question_parts
            if part.strip()
        ]

        if len(question_parts) >= 2:

            return [
                QueryAnalyzer._clean_sub_query(
                    part
                )
                for part in question_parts[:3]
            ]

        # ------------------------------------------------------
        # Safe fallback
        #
        # Do NOT create fake "first aspect" / "second
        # aspect" queries.
        # ------------------------------------------------------

        return [
            QueryAnalyzer._clean_sub_query(
                query
            )
        ]

    # ==========================================================
    # DOCUMENT REFERENCE EXTRACTION
    # ==========================================================

    @staticmethod
    def _extract_document_references(
        query: str,
        doc_keywords: list[str],
    ) -> list[str]:
        """
        Extract likely document references from a query.
        """

        references = []

        tokens = query.split()

        for token in tokens:

            clean_token = token.strip(
                "\"'.,!?;:()[]{}"
            )

            token_lower = (
                clean_token.lower()
            )

            if not clean_token:
                continue

            # Filename.
            if re.search(
                r"\.(?:txt|pdf|md|docx?|pptx?|xlsx?)$",
                token_lower,
                re.IGNORECASE,
            ):
                references.append(
                    clean_token
                )
                continue

            # Explicit document keyword.
            if any(
                keyword in token_lower
                for keyword in doc_keywords
            ):
                references.append(
                    clean_token
                )

        # Deduplicate while preserving order.
        unique = []
        seen = set()

        for reference in references:

            normalized = reference.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(reference)

        return unique

    # ==========================================================
    # SUB-QUERY CLEANING
    # ==========================================================

    @staticmethod
    def _clean_sub_query(
        query: str,
    ) -> str:
        """
        Normalize a generated sub-query.
        """

        query = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        # Remove punctuation immediately before '?'.
        query = re.sub(
            r"[,;:]+\s*\?$",
            "?",
            query,
        )

        # Remove trailing comma/semicolon/colon.
        query = re.sub(
            r"[,;:]+$",
            "",
            query,
        ).strip()

        if not query.endswith("?"):
            query += "?"

        return query

    # ==========================================================
    # FAKE SUB-QUERY DETECTION
    # ==========================================================

    @staticmethod
    def _is_fake_subquery(
        query: str,
    ) -> bool:
        """
        Detect meaningless LLM-generated fallback queries.
        """

        query_lower = query.lower().strip()

        fake_patterns = [
            "first aspect of:",
            "second aspect of:",
            "information regarding the first aspect",
            "information regarding the second aspect",
            "first topic in:",
            "second topic in:",
        ]

        return any(
            pattern in query_lower
            for pattern in fake_patterns
        )
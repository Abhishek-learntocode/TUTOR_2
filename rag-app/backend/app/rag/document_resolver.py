import re

from app.models.state import Message


class DocumentResolver:
    """
    Resolve user document references against the uploaded
    document catalog.

    Resolution hierarchy:

    1. Exact filename / document ID match
    2. Title / metadata match
    3. Explicit unknown filename check
    4. Conversation-context match
    5. Partial filename / document-type match
    6. Generic-term ambiguity detection
    """

    def __init__(self, llm=None):
        self.llm = llm

    # ==========================================================
    # DOCUMENT CATALOG
    # ==========================================================

    def get_catalog(
        self,
        vector_store,
    ) -> list[dict]:
        """
        Build a lightweight document catalog from the
        vector store.

        One catalog entry is created per document_id.
        """

        all_docs = vector_store.get_all_documents()

        catalog = []
        seen = set()

        for doc in all_docs:

            metadata = (
                getattr(
                    doc,
                    "metadata",
                    {},
                )
                or {}
            )

            document_id = (
                metadata.get("document_id")
                or metadata.get("source_filename")
            )

            source_filename = (
                metadata.get("source_filename")
                or document_id
            )

            document_type = (
                metadata.get(
                    "document_type",
                    "book",
                )
            )

            if not document_id:
                continue

            document_id = str(document_id)
            source_filename = str(source_filename)

            if document_id in seen:
                continue

            seen.add(document_id)

            clean_title = (
                re.sub(
                    r"\.[^.]+$",
                    "",
                    source_filename,
                )
                .replace("_", " ")
                .replace("-", " ")
                .strip()
                .lower()
            )

            catalog.append(
                {
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "document_type": str(
                        document_type
                    ),
                    "clean_title": clean_title,
                }
            )

        return catalog

    # ==========================================================
    # RESOLUTION
    # ==========================================================

    def resolve(
        self,
        query: str,
        catalog: list[dict],
        intent: str = "question",
        chat_history: list[Message] = None,
    ) -> tuple[list[str], list[str]]:
        """
        Resolve document references from a user query.

        Returns:

            (
                resolved_document_ids,
                ambiguous_candidates,
            )
        """

        if not catalog:
            return [], []

        query_lower = (
            query.lower().strip()
        )

        if not query_lower:
            return [], []

        # ------------------------------------------------------
        # STAGE 1
        # Exact filename / document ID match
        #
        # Example:
        # "What is in OS_Notes.pdf?"
        # ------------------------------------------------------

        exact_matches = []

        for document in catalog:

            filename = (
                document[
                    "source_filename"
                ].lower()
            )

            document_id = (
                document[
                    "document_id"
                ].lower()
            )

            if (
                filename in query_lower
                or document_id in query_lower
            ):
                exact_matches.append(
                    document[
                        "document_id"
                    ]
                )

        exact_matches = (
            self._unique(exact_matches)
        )

        if len(exact_matches) == 1:
            return exact_matches, []

        if len(exact_matches) > 1:
            return [], exact_matches

        # ------------------------------------------------------
        # STAGE 2
        # Title / metadata match
        #
        # Example:
        # "Explain OS Notes"
        # ------------------------------------------------------

        title_matches = []

        for document in catalog:

            title = document[
                "clean_title"
            ]

            if (
                title
                and title in query_lower
            ):
                title_matches.append(
                    document[
                        "document_id"
                    ]
                )

        title_matches = (
            self._unique(title_matches)
        )

        if len(title_matches) == 1:
            return title_matches, []

        if len(title_matches) > 1:
            return [], title_matches

        # ------------------------------------------------------
        # STAGE 3
        # Explicit filename with unsupported / unknown
        # extension.
        #
        # Example:
        # "Explain XYZ.xyz"
        #
        # If the user clearly supplied a filename that does
        # not exist in the catalog, don't fall through to a
        # generic keyword match.
        # ------------------------------------------------------

        explicit_filename = re.search(
            r"\b[\w\-. ]+\.[a-zA-Z0-9]{2,8}\b",
            query,
            re.IGNORECASE,
        )

        if explicit_filename:

            requested_filename = (
                explicit_filename
                .group(0)
                .strip()
                .lower()
            )

            known_filename = any(
                document[
                    "source_filename"
                ].lower()
                == requested_filename
                for document in catalog
            )

            if not known_filename:
                return [], []

        # ------------------------------------------------------
        # STAGE 4
        # Conversation context
        #
        # Examples:
        # "that document"
        # "the previous book"
        # "the notes"
        # ------------------------------------------------------

        vague_cues = [
            "that document",
            "that book",
            "the notes",
            "previous book",
            "previous document",
            "that file",
            "the file",
            "the book",
            "that",
        ]

        has_vague_reference = any(
            cue in query_lower
            for cue in vague_cues
        )

        if (
            chat_history
            and has_vague_reference
        ):

            conversation_matches = (
                self._resolve_from_history(
                    chat_history,
                    catalog,
                )
            )

            if len(conversation_matches) == 1:
                return conversation_matches, []

            if len(conversation_matches) > 1:
                return [], conversation_matches

        # ------------------------------------------------------
        # STAGE 5
        # Partial filename / document type
        #
        # Examples:
        # "OS notes"
        # "OS book"
        # "exam paper"
        # ------------------------------------------------------

        partial_matches = []

        for document in catalog:

            title = document[
                "clean_title"
            ]

            filename = (
                document[
                    "source_filename"
                ].lower()
            )

            document_type = (
                document[
                    "document_type"
                ].lower()
            )

            # ----------------------------------------------
            # Notes
            # ----------------------------------------------

            if (
                "notes" in query_lower
                or "note" in query_lower
            ):

                if (
                    "note" in title
                    or "note" in filename
                ):
                    partial_matches.append(
                        document[
                            "document_id"
                        ]
                    )

                    continue

            # ----------------------------------------------
            # Textbook / book
            # ----------------------------------------------

            if (
                "textbook"
                in query_lower
                or re.search(
                    r"\bbook\b",
                    query_lower,
                )
            ):

                if (
                    document_type
                    in {
                        "book",
                        "textbook",
                    }
                    or "textbook" in title
                    or "book" in filename
                ):
                    partial_matches.append(
                        document[
                            "document_id"
                        ]
                    )

                    continue

            # ----------------------------------------------
            # Exam / GATE
            # ----------------------------------------------

            if (
                "exam paper"
                in query_lower
                or re.search(
                    r"\bexam\b",
                    query_lower,
                )
                or re.search(
                    r"\bgate\b",
                    query_lower,
                )
            ):

                if (
                    document_type
                    == "exam_paper"
                    or "exam" in filename
                    or "gate" in title
                    or "gate" in filename
                ):
                    partial_matches.append(
                        document[
                            "document_id"
                        ]
                    )

                    continue

        partial_matches = (
            self._unique(partial_matches)
        )

        if len(partial_matches) == 1:
            return partial_matches, []

        if len(partial_matches) > 1:
            return [], partial_matches

        # ------------------------------------------------------
        # STAGE 6
        # Generic-term ambiguity
        #
        # Example:
        # "Summarise OS"
        #
        # If several OS-related documents exist,
        # explicitly ask the user to choose.
        # ------------------------------------------------------

        if self._contains_generic_os_term(
            query_lower
        ):

            os_documents = [
                document[
                    "document_id"
                ]
                for document in catalog
                if (
                    re.search(
                        r"\bos\b",
                        document[
                            "source_filename"
                        ].lower(),
                    )
                    or re.search(
                        r"\bos\b",
                        document[
                            "clean_title"
                        ].lower(),
                    )
                )
            ]

            os_documents = (
                self._unique(
                    os_documents
                )
            )

            if len(os_documents) > 1:
                return [], os_documents

            if len(os_documents) == 1:
                return os_documents, []

        # ------------------------------------------------------
        # No confident match
        # ------------------------------------------------------

        return [], []

    # ==========================================================
    # CONVERSATION RESOLUTION
    # ==========================================================

    @staticmethod
    def _resolve_from_history(
        chat_history: list[Message],
        catalog: list[dict],
    ) -> list[str]:
        """
        Find document references mentioned in recent
        conversation history.

        The most recent relevant document mention wins.
        """

        for message in reversed(
            chat_history
        ):

            content = (
                getattr(
                    message,
                    "content",
                    "",
                )
                or ""
            )

            content_lower = (
                content.lower()
            )

            matches = []

            for document in catalog:

                filename = (
                    document[
                        "source_filename"
                    ].lower()
                )

                title = document[
                    "clean_title"
                ]

                document_id = (
                    document[
                        "document_id"
                    ].lower()
                )

                if (
                    filename in content_lower
                    or (
                        title
                        and title in content_lower
                    )
                    or (
                        document_id
                        in content_lower
                    )
                ):
                    matches.append(
                        document[
                            "document_id"
                        ]
                    )

            matches = (
                DocumentResolver._unique(
                    matches
                )
            )

            if matches:
                return matches

        return []

    # ==========================================================
    # GENERIC TERM DETECTION
    # ==========================================================

    @staticmethod
    def _contains_generic_os_term(
        query_lower: str,
    ) -> bool:
        """
        Detect 'OS' as a standalone generic term.

        Avoids matching words such as:
            "those"
            "most"
            "post"
        """

        return bool(
            re.search(
                r"\bos\b",
                query_lower,
            )
        )

    # ==========================================================
    # UTILITY
    # ==========================================================

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:
        """
        Deduplicate while preserving order.
        """

        result = []
        seen = set()

        for value in values:

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result
import time
import requests


class SemanticAPI:

    BASE = (
        "https://api.semanticscholar.org"
        "/graph/v1"
    )

    def __init__(
        self,
        api_key=None,
        retry=999999,
        retry_sleep=2,
    ):

        self.retry = retry
        self.retry_sleep = retry_sleep

        self.session = (
            requests.Session()
        )

        if api_key:

            self.session.headers.update(
                {
                    "x-api-key": api_key
                }
            )

    # --------------------
    # common request
    # --------------------

    def _get(
        self,
        endpoint,
        params=None,
    ):

        retry = 0

        while True:

            try:

                r = self.session.get(
                    self.BASE + endpoint,
                    params=params,
                    timeout=120,
                )

                if r.status_code == 429:

                    time.sleep(
                        self.retry_sleep
                    )

                    retry += 1

                    continue

                r.raise_for_status()

                return r.json()

            except Exception as e:

                retry += 1

                if retry > self.retry:

                    raise e

                print(
                    f"retry={retry}",
                    e,
                )

                time.sleep(
                    self.retry_sleep
                )

    # --------------------
    # paper bulk
    # --------------------

    def get_papers(
        self,
        *,
        field,
        year,
        min_citation=1,
        publication_type="JournalArticle",
        token=None,
        require_abstract=True,
    ):

        params = {
            "fieldsOfStudy": field,
            "year": year,
            "minCitationCount": min_citation,
            "publicationTypes": publication_type,
            "sort": "citationCount:desc",
            "fields": ",".join(
                [
                    "title",
                    "abstract",
                    "url",
                    "venue",
                    "year",
                    "journal",
                    "publicationDate",
                    "referenceCount",
                    "citationCount",
                    "externalIds",
                    "fieldsOfStudy",
                    "s2FieldsOfStudy",
                    "publicationTypes",
                ]
            ),
        }

        if token:
            params[
                "token"
            ] = token

        result = self._get(
            "/paper/search/bulk",
            params,
        )

        papers = (
            result.get(
                "data"
            )
            or []
        )

        if require_abstract:

            papers = [
                p
                for p in papers
                if (
                    p.get(
                        "abstract"
                    )
                    and p[
                        "abstract"
                    ].strip()
                )
            ]

        return {
            "total":
            result.get(
                "total"
            ),

            "token":
            result.get(
                "token"
            ),

            "papers":
            papers,
        }

    # --------------------
    # author
    # --------------------

    def get_author(
        self,
        author_id,
    ):

        return self._get(
            f"/author/{author_id}"
        )

    # --------------------
    # citations
    # --------------------

    def get_citations(
        self,
        corpus_id,
        limit=1000,
    ):

        return self._get(
            f"/paper/{corpus_id}/citations",
            {
                "limit": limit
            },
        )

    # --------------------
    # references
    # --------------------

    def get_references(
        self,
        corpus_id,
        limit=1000,
    ):

        return self._get(
            f"/paper/{corpus_id}/references",
            {
                "limit": limit
            },
        )
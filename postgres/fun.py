import gzip
from tqdm import tqdm
from postgres.model import Paper, Field, PaperField, PublicationType, PaperPublicationType, PaperAuthor, Author, Citation, Embedding
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects.postgresql import insert

import json
from datetime import datetime



def parse_date(value):
    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d"
        ).date()
    except Exception:
        return None
    

# -----------

def import_papers(
    engine,
    gz_path="../opt/papers.gz",
    batch_size=5000
):
    field_set = set()

    publication_type_set = set()

    paper_rows = []

    field_rows = []

    paper_field_rows = []

    publication_type_rows = []

    paper_publication_type_rows = []

    paper_author_rows = []

    with Session(engine) as session:

        with gzip.open(
            gz_path,
            "rt",
            encoding="utf-8"
        ) as f:

            for line in tqdm(
                f,
                desc="Import Papers"
            ):

                paper = json.loads(line)

                corpus_id = paper.get(
                    "corpusid"
                )

                if not corpus_id:
                    continue

                external_ids = (
                    paper.get(
                        "externalids"
                    )
                    or {}
                )

                journal = (
                    paper.get(
                        "journal"
                    )
                    or {}
                )

                # ==================================================
                # Paper
                # ==================================================

                paper_rows.append(
                    {
                        "corpus_id": corpus_id,
                        "title": paper.get(
                            "title"
                        ),
                        "url": paper.get(
                            "url"
                        ),
                        "venue": paper.get(
                            "venue"
                        ),
                        "publication_venue_id": paper.get(
                            "publicationvenueid"
                        ),
                        "year": paper.get(
                            "year"
                        ),
                        "publication_date": parse_date(
                            paper.get(
                                "publicationdate"
                            )
                        ),
                        "reference_count": paper.get(
                            "referencecount",
                            0
                        ),
                        "citation_count": paper.get(
                            "citationcount",
                            0
                        ),
                        "influential_citation_count": paper.get(
                            "influentialcitationcount",
                            0
                        ),
                        "is_open_access": paper.get(
                            "isopenaccess",
                            False
                        ),

                        "doi": external_ids.get(
                            "DOI"
                        ),

                        "pubmed": external_ids.get(
                            "PubMed"
                        ),

                        "mag": external_ids.get(
                            "MAG"
                        ),

                        "dblp": external_ids.get(
                            "DBLP"
                        ),

                        "arxiv": external_ids.get(
                            "ArXiv"
                        ),

                        "acl": external_ids.get(
                            "ACL"
                        ),

                        "corpus_external_id": external_ids.get(
                            "CorpusId"
                        ),

                        "external_ids": external_ids,

                        "open_access_info": paper.get(
                            "openaccessinfo"
                        ),

                        "journal_name": journal.get(
                            "name"
                        ),

                        "journal_volume": journal.get(
                            "volume"
                        ),

                        "journal_pages": journal.get(
                            "pages"
                        ),
                    }
                )

                # ==================================================
                # Field
                # ==================================================

                for field in (
                    paper.get(
                        "s2fieldsofstudy"
                    )
                    or []
                ):

                    category = field.get(
                        "category"
                    )

                    source = field.get(
                        "source"
                    )

                    if not category:
                        continue

                    if category not in field_set:

                        field_set.add(
                            category
                        )

                        field_rows.append(
                            {
                                "category": category
                            }
                        )

                    paper_field_rows.append(
                        {
                            "paper_id": corpus_id,
                            "category": category,
                            "source": source,
                        }
                    )

                # ==================================================
                # PublicationType
                # ==================================================

                for pub_type in (
                    paper.get(
                        "publicationtypes"
                    )
                    or []
                ):

                    if (
                        pub_type
                        not in publication_type_set
                    ):

                        publication_type_set.add(
                            pub_type
                        )

                        publication_type_rows.append(
                            {
                                "name": pub_type
                            }
                        )

                    paper_publication_type_rows.append(
                        {
                            "paper_id": corpus_id,
                            "publication_type_name": pub_type,
                        }
                    )

                # ==================================================
                # Batch Flush
                # ==================================================

                if len(paper_rows) >= batch_size:

                    session.bulk_insert_mappings(
                        Paper,
                        paper_rows
                    )

                    if field_rows:

                        session.bulk_insert_mappings(
                            Field,
                            field_rows
                        )

                    if paper_field_rows:

                        session.bulk_insert_mappings(
                            PaperField,
                            paper_field_rows
                        )

                    if publication_type_rows:

                        session.bulk_insert_mappings(
                            PublicationType,
                            publication_type_rows
                        )

                    if paper_publication_type_rows:

                        session.bulk_insert_mappings(
                            PaperPublicationType,
                            paper_publication_type_rows
                        )

                    if paper_author_rows:

                        session.bulk_insert_mappings(
                            PaperAuthor,
                            paper_author_rows
                        )

                    session.commit()

                    paper_rows.clear()

                    field_rows.clear()

                    paper_field_rows.clear()

                    publication_type_rows.clear()

                    paper_publication_type_rows.clear()

                    paper_author_rows.clear()

            # ==================================================
            # Final Flush
            # ==================================================

            if paper_rows:

                session.bulk_insert_mappings(
                    Paper,
                    paper_rows
                )

                if field_rows:

                    session.bulk_insert_mappings(
                        Field,
                        field_rows
                    )

                if paper_field_rows:

                    session.bulk_insert_mappings(
                        PaperField,
                        paper_field_rows
                    )

                if publication_type_rows:

                    session.bulk_insert_mappings(
                        PublicationType,
                        publication_type_rows
                    )

                if paper_publication_type_rows:

                    session.bulk_insert_mappings(
                        PaperPublicationType,
                        paper_publication_type_rows
                    )

                if paper_author_rows:

                    session.bulk_insert_mappings(
                        PaperAuthor,
                        paper_author_rows
                    )

                session.commit()

    print("Import papers completed.")
# -------------

def import_authors(
    engine,
    path="../opt/authors.gz",
    batch_size=50000,
):

    rows = []

    total = 0

    with Session(engine) as session:

        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
        ) as f:

            for line in tqdm(
                f,
                desc="Import Authors"
            ):

                try:

                    obj = json.loads(
                        line
                    )

                    author_id = obj.get(
                        "authorid"
                    )

                    if not author_id:
                        continue

                    rows.append(
                        {
                            "author_id": int(
                                author_id
                            ),

                            "name": obj.get(
                                "name"
                            ),

                            "aliases": obj.get(
                                "aliases"
                            ),

                            "affiliations": obj.get(
                                "affiliations"
                            ),

                            "homepage": obj.get(
                                "homepage"
                            ),

                            "url": obj.get(
                                "url"
                            ),

                            "paper_count": obj.get(
                                "papercount"
                            ),

                            "citation_count": obj.get(
                                "citationcount"
                            ),

                            "h_index": obj.get(
                                "hindex"
                            ),

                            "external_ids": obj.get(
                                "externalids"
                            ),
                        }
                    )

                    if (
                        len(rows)
                        >= batch_size
                    ):

                        session.bulk_insert_mappings(
                            Author,
                            rows,
                        )

                        session.commit()

                        total += len(
                            rows
                        )

                        rows.clear()

                except Exception as e:

                    print(
                        e
                    )

        if rows:

            session.bulk_insert_mappings(
                Author,
                rows,
            )

            session.commit()

            total += len(
                rows
            )

    print(
        f"Imported {total:,} authors"
    )


# -------?
def import_citations(
    engine,
    path="../opt/citations.gz",
    batch_size=5000,
):

    rows=[]

    with Session(engine) as session:

        for line in tqdm(
            gzip.open(
                path,
                "rt",
                encoding="utf8"
            ),
            desc="Import Citation",
        ):

            try:

                obj=json.loads(
                    line
                )

                citing=obj.get(
                    "citingcorpusid"
                )

                cited=obj.get(
                    "citedcorpusid"
                )

                if (
                    not citing
                    or not cited
                ):
                    continue

                rows.append(
                    {
                        "citation_id":
                            obj.get(
                                "citationid"
                            ),

                        "citing_corpus_id":
                            citing,

                        "cited_corpus_id":
                            cited,

                        "is_influential":
                            obj.get(
                                "isinfluential",
                                False
                            ),

                        "contexts":
                            obj.get(
                                "contexts"
                            ),

                        "intents":
                            obj.get(
                                "intents"
                            ),
                    }
                )

                if len(rows)>=batch_size:

                    flush_citation(
                        session,
                        rows,
                    )

            except Exception:

                continue

        if rows:

            flush_citation(
                session,
                rows,
            )

    print(
        "Citation import completed."
    )


def flush_citation(
    session,
    rows,
):

    ids=set()

    for r in rows:

        ids.add(
            r[
                "citing_corpus_id"
            ]
        )

        ids.add(
            r[
                "cited_corpus_id"
            ]
        )

    existing=set(
        session.scalars(
            select(
                Paper.corpus_id
            ).where(
                Paper.corpus_id.in_(
                    ids
                )
            )
        )
    )

    valid=[]

    for r in rows:

        if (
            r[
                "citing_corpus_id"
            ]
            in existing
            and
            r[
                "cited_corpus_id"
            ]
            in existing
        ):

            valid.append(
                r
            )

    if valid:

        session.execute(
            insert(
                Citation
            )
            .values(
                valid
            )
            .on_conflict_do_nothing()
        )

        session.commit()

    rows.clear()


    # --------------
def import_embedding(
    engine,
    gz_path,
    batch_size=2000,
):
    session = Session(engine)

    rows = []

    count = 0
    skipped = 0

    with gzip.open(
        gz_path,
        "rt",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Import Embeddings",
        ):

            try:

                item = json.loads(line)

                corpus_id = item["corpusid"]

                # paper必须存在
                exists = session.get(
                    Paper,
                    corpus_id,
                )

                if not exists:
                    skipped += 1
                    continue

                vector = json.loads(
                    item["vector"]
                )

                if len(vector) != 768:
                    skipped += 1
                    continue

                rows.append(
                    {
                        "corpus_id": corpus_id,
                        "model": item["model"],
                        "vector": vector,
                    }
                )

                if len(rows) >= batch_size:

                    stmt = (
                        insert(
                            Embedding
                        )
                        .values(rows)
                        .on_conflict_do_update(
                            index_elements=[
                                Embedding.corpus_id,
                                Embedding.model,
                            ],
                            set_={
                                "vector":
                                insert(
                                    Embedding
                                ).excluded.vector
                            },
                        )
                    )

                    session.execute(
                        stmt
                    )

                    session.commit()

                    count += len(rows)

                    rows.clear()

            except Exception as e:

                skipped += 1

                if skipped % 1000 == 0:
                    print(
                        "skip:",
                        e
                    )

        if rows:

            stmt = (
                insert(
                    Embedding
                )
                .values(rows)
                .on_conflict_do_update(
                    index_elements=[
                        Embedding.corpus_id,
                        Embedding.model,
                    ],
                    set_={
                        "vector":
                        insert(
                            Embedding
                        ).excluded.vector
                    },
                )
            )

            session.execute(
                stmt
            )

            session.commit()

            count += len(rows)

    session.close()

    print(
        f"Imported={count}"
    )

    print(
        f"Skipped={skipped}"
    )

    return True


# ----------------


import gzip
import json

from tqdm import tqdm

from sqlalchemy import (
    text,
)
from sqlalchemy.orm import (
    Session,
)


def import_abstract(
    engine,
    gz_path,
    batch_size=5000,
):

    session = Session(engine)

    batch = []

    updated = 0
    skipped = 0
    empty = 0

    sql = text("""
        UPDATE paper
        SET
            abstract = :abstract
        WHERE corpus_id = :corpus_id
    """)

    try:

        with gzip.open(
            gz_path,
            "rt",
            encoding="utf-8",
        ) as f:

            for line in tqdm(
                f,
                desc="Import Abstract",
            ):

                try:

                    item = json.loads(
                        line
                    )

                    abstract = item.get(
                        "abstract"
                    )


                    # 没有内容直接跳过
                    if (
                        not abstract
                    ):
                        empty += 1
                        continue

                    batch.append(
                        {
                            "corpus_id":
                            item[
                                "corpusid"
                            ],

                            "abstract":
                            abstract
                        }
                    )

                    if (
                        len(batch)
                        >= batch_size
                    ):

                        result = (
                            session.execute(
                                sql,
                                batch,
                            )
                        )

                        session.commit()

                        updated += (
                            result.rowcount
                            or 0
                        )

                        batch.clear()

                except Exception as e:

                    session.rollback()

                    skipped += 1

                    if (
                        skipped
                        % 100
                        == 0
                    ):
                        print(e)

            if batch:

                result = (
                    session.execute(
                        sql,
                        batch,
                    )
                )

                session.commit()

                updated += (
                    result.rowcount
                    or 0
                )

    finally:

        session.close()

    print()

    print(
        f"Actual Updated: {updated}"
    )

    print(
        f"Empty: {empty}"
    )

    print(
        f"Skipped: {skipped}"
    )

    return True

# -------------------

def import_missing_authors_and_relations(
    engine,
    path="../opt/papers.gz",
    batch_size=10000,
):

    author_rows=[]

    relation_rows=[]

    with Session(engine) as session:

        with gzip.open(
            path,
            "rt",
            encoding="utf8",
        ) as f:

            for line in tqdm(
                f,
                desc="Author Relation"
            ):

                paper=json.loads(
                    line
                )

                paper_id=paper.get(
                    "corpusid"
                )

                if not paper_id:
                    continue

                seen=set()

                for order, author in enumerate(
                    paper.get(
                        "authors"
                    )
                    or []
                ):

                    aid=author.get(
                        "authorId"
                    )

                    if not aid:
                        continue

                    aid=int(
                        aid
                    )

                    key=(
                        paper_id,
                        aid,
                    )

                    if key in seen:
                        continue

                    seen.add(
                        key
                    )

                    author_rows.append(
                        {
                            "author_id": aid,

                            "name": author.get(
                                "name"
                            ),
                        }
                    )

                    relation_rows.append(
                        {
                            "paper_id": paper_id,

                            "author_id": aid,

                            "author_order": order,
                        }
                    )

                if (
                    len(
                        relation_rows
                    )
                    >= batch_size
                ):

                    session.execute(
                        insert(
                            Author
                        )
                        .values(
                            author_rows
                        )
                        .on_conflict_do_nothing()
                    )

                    session.execute(
                        insert(
                            PaperAuthor
                        )
                        .values(
                            relation_rows
                        )
                        .on_conflict_do_nothing()
                    )

                    session.commit()

                    author_rows.clear()

                    relation_rows.clear()

        if author_rows:

            session.execute(
                insert(
                    Author
                )
                .values(
                    author_rows
                )
                .on_conflict_do_nothing()
            )

        if relation_rows:

            session.execute(
                insert(
                    PaperAuthor
                )
                .values(
                    relation_rows
                )
                .on_conflict_do_nothing()
            )

        session.commit()

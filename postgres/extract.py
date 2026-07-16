import pandas as pd
from sqlalchemy import (
    func,
    literal,
    select,
)
from sqlalchemy.orm import Session

from postgres.model import (
    Paper,
    PaperField,
    PaperPublicationType,
)

fields = [
    'Computer Science',
    'Engineering',
    'Physics',
    'Chemistry',
    'Biology',
    'Medicine',
    'Economics',
    'Psychology'
]


SEED = "42"

TYPE = "JournalArticle"
YEAR = 2024

LOW_N = 60
MID_N = 100
HIGH_N = 40


def get_seed_papers(
    engine,
    field,
    year=YEAR,
    type=TYPE,
    low_n=LOW_N,
    mid_n=MID_N,
    high_n=HIGH_N,
):

    with Session(engine) as session:

        # Step1 去重
        paper_base = (

            select(

                Paper.corpus_id.label(
                    "id"
                ),

                Paper.doi,
                Paper.abstract,
                Paper.journal_name,

                func.max(
                    Paper.title
                ).label(
                    "title"
                ),

                func.max(
                    Paper.year
                ).label(
                    "year"
                ),

                func.max(
                    Paper.citation_count
                ).label(
                    "citation_count"
                )

            )

            .join(
                PaperField,
                Paper.corpus_id
                ==
                PaperField.paper_id
            )

            .join(
                PaperPublicationType,
                Paper.corpus_id
                ==
                PaperPublicationType.paper_id
            )

            .where(

                Paper.year
                ==
                year,

                Paper.doi!='',

                Paper.citation_count
                >
                0,

                PaperField.category
                ==
                field,

                PaperField.source
                ==
                "s2-fos-model",

                PaperPublicationType.publication_type_name
                ==
                type,
            )

            .group_by(
                Paper.corpus_id
            )

            .having(
                func.count(
                    PaperField.paper_id
                )
                ==
                1
            )

            .subquery()
        )

        # Step2 分桶
        ranked = (

            select(

                paper_base.c.id,

                paper_base.c.doi,

                paper_base.c.title,

                paper_base.c.abstract,

                paper_base.c.journal_name,

                paper_base.c.year,

                paper_base.c.citation_count,

                func.ntile(10)
                .over(
                    order_by=
                    paper_base
                    .c
                    .citation_count
                )
                .label(
                    "bucket"
                )

            )

            .subquery()
        )

        result = []

        groups = [

            (
                "low",
                range(1, 5),
                low_n
            ),

            (
                "mid",
                range(5, 10),
                mid_n
            ),

            (
                "high",
                range(10, 11),
                high_n
            ),

        ]

        for name, buckets, limit in groups:

            q = (

                select(
                    ranked
                    .c
                    .id,

                    ranked.c.doi,

                    ranked
                    .c
                    .title,

                    ranked.c.abstract,
                    ranked.c.journal_name,

                    ranked
                    .c
                    .year,

                    ranked
                    .c
                    .citation_count,
                )

                .where(
                    ranked
                    .c
                    .bucket
                    .in_(
                        buckets
                    )
                )

                .order_by(

                    func.md5(

                        func.concat(

                            ranked
                            .c
                            .id,

                            literal(
                                SEED
                            )
                        )
                    )
                )

                .limit(
                    limit
                )

            )

            rows = (
                session
                .execute(q)
                .mappings()
                .all()
            )

            for r in rows:

                result.append({

                    **r,

                    "group": name,

                    "field": field

                })

        df = pd.DataFrame(
            result
        )

        return (
            df
            .sort_values(
                [
                    "group",
                    "citation_count"
                ]
            )
            .reset_index(
                drop=True
            )
        )
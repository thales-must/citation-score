from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)


class Base(DeclarativeBase):
    pass


# =====================================================
# Paper
# =====================================================

class Paper(Base):
    __tablename__ = "paper"

    corpus_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    abstract: Mapped[str | None] = mapped_column(
        Text
    )

    url: Mapped[str | None] = mapped_column(
        Text
    )

    venue: Mapped[str | None] = mapped_column(
        Text
    )

    publication_venue_id: Mapped[str | None] = mapped_column(
        String(128)
    )

    year: Mapped[int | None] = mapped_column(
        Integer,
        index=True
    )

    publication_date: Mapped[date | None] = mapped_column(
        Date
    )

    reference_count: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    citation_count: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    influential_citation_count: Mapped[int] = mapped_column(
        Integer,
        default=0
    )

    is_open_access: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )

    # --------------------------------
    # searchable external ids
    # --------------------------------

    doi: Mapped[str | None] = mapped_column(
        String(255),
        index=True
    )

    pubmed: Mapped[str | None] = mapped_column(
        String(100),
        index=True
    )

    mag: Mapped[str | None] = mapped_column(
        String(100),
        index=True
    )

    dblp: Mapped[str | None] = mapped_column(
        String(255),
        index=True
    )

    arxiv: Mapped[str | None] = mapped_column(
        String(100),
        index=True
    )

    acl: Mapped[str | None] = mapped_column(
        String(100),
        index=True
    )

    corpus_external_id: Mapped[str | None] = mapped_column(
        String(100)
    )

    external_ids: Mapped[dict | None] = mapped_column(
        JSONB
    )

    # --------------------------------
    # open access
    # --------------------------------

    open_access_info: Mapped[dict | None] = mapped_column(
        JSONB
    )

    # --------------------------------
    # journal
    # --------------------------------

    journal_name: Mapped[str | None] = mapped_column(
        Text
    )

    journal_volume: Mapped[str | None] = mapped_column(
        String(100)
    )

    journal_pages: Mapped[str | None] = mapped_column(
        String(255)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )


# =====================================================
# Author
# =====================================================

class Author(Base):
    __tablename__ = "author"

    author_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True
    )

    aliases: Mapped[list | None] = mapped_column(
        JSONB
    )

    affiliations: Mapped[list | None] = mapped_column(
        JSONB
    )

    homepage: Mapped[str | None] = mapped_column(
        Text
    )

    url: Mapped[str | None] = mapped_column(
        Text
    )

    paper_count: Mapped[int | None] = mapped_column(
        Integer
    )

    citation_count: Mapped[int | None] = mapped_column(
        Integer
    )

    h_index: Mapped[int | None] = mapped_column(
        Integer
    )

    external_ids: Mapped[dict | None] = mapped_column(
        JSONB
    )


# =====================================================
# PaperAuthor
# =====================================================

class PaperAuthor(Base):
    __tablename__ = "paper_author"

    paper_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "paper.corpus_id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    author_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "author.author_id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    author_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )


# =====================================================
# Field
# =====================================================

class Field(Base):
    __tablename__ = "field"

    category: Mapped[str] = mapped_column(
        String(100),
        primary_key=True
    )


# =====================================================
# PaperField
# =====================================================

class PaperField(Base):
    __tablename__ = "paper_field"

    paper_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "paper.corpus_id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    category: Mapped[str] = mapped_column(
        String(100),
        ForeignKey(
            "field.category",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    source: Mapped[str] = mapped_column(
        String(50),
        primary_key=True
    )


# =====================================================
# PublicationType
# =====================================================

class PublicationType(Base):
    __tablename__ = "publication_type"

    name: Mapped[str] = mapped_column(
        String(100),
        primary_key=True
    )


# =====================================================
# PaperPublicationType
# =====================================================

class PaperPublicationType(Base):
    __tablename__ = "paper_publication_type"

    paper_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "paper.corpus_id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    publication_type_name: Mapped[str] = mapped_column(
        String(100),
        ForeignKey(
            "publication_type.name",
            ondelete="CASCADE"
        ),
        primary_key=True
    )


# =====================================================
# Citation
# =====================================================

class Citation(Base):
    __tablename__ = "citation"

    citing_corpus_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "paper.corpus_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        nullable=False,
    )

    cited_corpus_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "paper.corpus_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
        nullable=False,
    )

    is_influential: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )

    contexts: Mapped[list | None] = mapped_column(
        JSONB
    )

    intents: Mapped[list | None] = mapped_column(
        JSONB
    )

    minilm_score: Mapped[float | None] = mapped_column(Float)

    specter_score: Mapped[float | None] = mapped_column(Float)

    sparse_score: Mapped[float | None] = mapped_column(Float)

    cross_score: Mapped[float | None] = mapped_column(Float)

    llm_score: Mapped[float | None] = mapped_column(Float)

    abstract_score: Mapped[float | None] = mapped_column(Float)

    abstract_cross_score: Mapped[float | None] = mapped_column(Float)
    
    abstract_llm_score: Mapped[float | None] = mapped_column(Float)

    author_independence: Mapped[float | None] = mapped_column(Float)

    scored_at: Mapped[datetime | None] = mapped_column(DateTime)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
    )


# =====================================================
# Embedding
# =====================================================

class Embedding(Base):
    __tablename__ = "embedding"

    corpus_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "paper.corpus_id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    model: Mapped[str] = mapped_column(
        String(100),
        primary_key=True
    )

    vector: Mapped[list] = mapped_column(
        Vector(768)
    )


# =====================================================
# Index
# =====================================================

Index(
    "idx_paper_citation_count",
    Paper.citation_count
)

Index(
    "idx_paper_year",
    Paper.year
)

Index(
    "idx_paper_doi",
    Paper.doi
)

Index(
    "idx_paper_pubmed",
    Paper.pubmed
)

Index(
    "idx_paper_mag",
    Paper.mag
)

Index(
    "idx_paper_arxiv",
    Paper.arxiv
)

Index(
    "idx_paper_acl",
    Paper.acl
)

Index(
    "idx_paper_field_category",
    PaperField.category
)

Index(
    "idx_paper_field_category_source",
    PaperField.category,
    PaperField.source
)

Index(
    "idx_paper_publication_type",
    PaperPublicationType.publication_type_name
)

Index(
    "idx_paper_author_author",
    PaperAuthor.author_id
)

Index(
    "idx_paper_author_paper",
    PaperAuthor.paper_id
)

Index(
    "idx_citation_citing",
    Citation.citing_corpus_id
)

Index(
    "idx_citation_cited",
    Citation.cited_corpus_id
)

Index(
    "idx_embedding_model",
    Embedding.model
)
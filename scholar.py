import math
import os
from collections import deque
from datetime import date, datetime
from time import sleep, time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Re-export for existing notebooks; standalone scorers can import the
# lightweight citation_prompts module without loading models or plotting tools.
from citation_prompts import (
    SYSTEM_PROMPT, build_abstract_prompt, build_context_prompt,
    build_prompt, build_scoring_prompt,
)

ContextInput = Union[str, Sequence[str]]
device = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

load_dotenv()

def get_session():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASS")
    db = os.getenv("POSTGRES_DB")

    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@localhost/{db}"
    )

    return Session(engine)

def get_corpus_id(paper: Dict[str, Any]) -> Optional[int]:
    """
    从 Semantic Scholar paper 对象中提取 corpus_id。

    兼容两种情况：
    1. paper["externalIds"]["CorpusId"]
    2. paper["corpusId"]
    """
    if not paper:
        return None

    external_ids = paper.get("externalIds") or {}

    corpus_id = (
        external_ids.get("CorpusId")
        or paper.get("corpusId")
        or paper.get("corpus_id")
    )

    if not corpus_id:
        return None

    return int(corpus_id)

def pd_clean(value: Any) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    text = " ".join(text.split())

    return text or None

def parse_date(value: Any) -> Optional[date]:
    if not value:
        return None

    try:
        return date.fromisoformat(str(value))
    except Exception:
        return None

class S2Api:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    API_KEY = os.getenv("API_KEY")
    RETRY_DURATION = 2
    LIMIT_DELAY = 5
    ERR_RETRY_COUNT = 10
    _retry_n = 0
    # _author_papers: Dict[str, List['IPaper']] = {}
    __cache: Dict[str, Dict[str, Dict]] = {}  # 双下划线实现名称重整，增强封装性
    __cache_expire: int = 36000  # 10小时过期

    def __init__(self) -> None:
        pass

    @classmethod
    def _get_from_cache(cls, cache_type: str, key: str) -> Optional[Any]:
        """私有方法：从缓存获取数据"""
        # 检查缓存类型是否存在
        if cache_type not in cls.__cache:
            return None

        # 检查键是否存在
        if key not in cls.__cache[cache_type]:
            return None

        cache_entry = cls.__cache[cache_type][key]

        # 检查是否过期
        current_time = time()
        if current_time - cache_entry["timestamp"] > cache_entry["ttl"]:
            # 缓存过期，删除
            del cls.__cache[cache_type][key]
            return None

        return cache_entry["data"]

    @classmethod
    def _save_to_cache(cls, cache_type: str, key: str, data: Any, ttl: int = None):
        """私有方法：保存数据到缓存"""
        # 确保缓存类型存在
        if cache_type not in cls.__cache:
            cls.__cache[cache_type] = {}

        # 使用配置的TTL或传入的TTL
        if ttl is None:
            ttl = cls.__cache_expire

        cls.__cache[cache_type][key] = {"data": data, "timestamp": time(), "ttl": ttl}

    @classmethod
    def clear_cache(cls, cache_type: str = None):
        """公开方法：清理缓存（对外提供清理接口）"""
        if cache_type:
            if cache_type in cls.__cache:
                cls.__cache[cache_type].clear()
                print(f"已清理 {cache_type} 缓存")
        else:
            cls.__cache.clear()
            print("已清理所有缓存")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """公开方法：获取缓存统计信息"""
        stats = {}
        for cache_type, cache in cls.__cache.items():
            stats[cache_type] = len(cache)
        return stats

    @classmethod
    def _get_cache_hit_rate(cls) -> Dict[str, Dict[str, int]]:
        """私有方法：内部使用的缓存命中率统计"""
        # 这里可以添加更详细的统计逻辑
        return {
            "summary": {
                "total_caches": len(cls.__cache),
                "total_entries": sum(len(cache) for cache in cls.__cache.values()),
            }
        }

    def paper_detail(self, id: str, p={}) -> Optional[Dict]:

        url = f"{self.BASE_URL}/paper/{id}"
        params = {
            "fields": "paperId,paperId,title,abstract,authors.name,authors.affiliations,year,publicationDate,venue,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,url,openAccessPdf,externalIds,corpusId,publicationDate,isOpenAccess,publicationTypes,journal,citationStyles,embedding.specter_v2,tldr,citations.title,citations.abstract", **p
        }
        try:
            resp = requests.get(url, params=params, headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                self.retry_n = 0
                data = resp.json()
                return data
            elif resp.status_code == 429:
                sleep(self.RETRY_DURATION)
                return self.paper_detail(id)
            else:
                if self.retry_n > self.ERR_RETRY_COUNT:
                    print(f"API请求失败: {resp.status_code} - {resp.text}")
                    self.retry_n = 0
                    return None
                else:
                    sleep(self.RETRY_DURATION)
                    self.retry_n += 1
                    return self.paper_detail(id)
        except requests.exceptions.RequestException as e:
            if self.retry_n > self.ERR_RETRY_COUNT:
                print(f"API请求异常: {e}")
                self.retry_n = 0
                return None
            else:
                sleep(self.RETRY_DURATION)
                self.retry_n += 1
                return self.paper_detail(id)
        finally:
            sleep(self.LIMIT_DELAY)  # 尊重API限制

    def paper_citations(self, id, p={}) -> Optional[List[Dict]]:
        url = f"{self.BASE_URL}/paper/{id}/citations"
        params = {"fields": "title,abstract,contexts,intents,isInfluential,authors,externalIds", 'limit':1000, **p}
        
        try:
            resp = requests.get(url, params=params, headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                self.retry_n = 0
                data = resp.json()["data"] or []
                return data
            elif resp.status_code == 429:
                return self.paper_citations(id, params)
            else:
                if self.retry_n > self.ERR_RETRY_COUNT:
                    print(f"API请求失败: {resp.status_code} - {resp.text}")
                    self.retry_n = 0
                    return None
                else:
                    sleep(self.RETRY_DURATION)
                    self.retry_n += 1
                    return self.paper_citations(id, params)
        except requests.exceptions.RequestException as e:
            if self.retry_n > self.ERR_RETRY_COUNT:
                print(f"API请求异常: {e}")
                self.retry_n = 0
                return None
            else:
                sleep(self.RETRY_DURATION)
                self.retry_n += 1
                return self.paper_citations(id, params)
        finally:
            sleep(self.LIMIT_DELAY)  # 尊重API限制

    def author_papers(self, id: str, p={}) -> Optional[List[Dict]]:
        url = f"{self.BASE_URL}/author/{id}/papers"
        params = {"fields": "title,year,publicationDate", 'limit':1000, **p}
        try:
            resp = requests.get(url, params=params, headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                self.retry_n = 0
                data = resp.json()["data"] or []
                return data
            elif resp.status_code == 429:
                sleep(self.RETRY_DURATION)
                return self.author_papers(id)
            else:
                if self.retry_n > self.ERR_RETRY_COUNT:
                    print(f"API请求失败: {resp.status_code} - {resp.text}")
                    self.retry_n = 0
                    return None
                else:
                    sleep(self.RETRY_DURATION)
                    self.retry_n += 1
                    return self.author_papers(id)
        except requests.exceptions.RequestException as e:
            if self.retry_n > self.ERR_RETRY_COUNT:
                print(f"API请求异常: {e}")
                self.retry_n = 0
                return None
            else:
                sleep(self.RETRY_DURATION)
                self.retry_n += 1
                return self.author_papers(id)
        finally:
            sleep(self.LIMIT_DELAY)  # 尊重API限制

    def author_detail(self, id: str, p={}) -> Optional[Dict]:

        url = f"{self.BASE_URL}/author/{id}"
        try:
            resp = requests.get(url,headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                self.retry_n = 0
                data = resp.json()
                return data
            elif resp.status_code == 429:
                sleep(self.RETRY_DURATION)
                return self.author_detail(id)
            else:
                if self.retry_n > self.ERR_RETRY_COUNT:
                    print(f"API请求失败: {resp.status_code} - {resp.text}")
                    self.retry_n = 0
                    return None
                else:
                    sleep(self.RETRY_DURATION)
                    self.retry_n += 1
                    return self.author_detail(id)
        except requests.exceptions.RequestException as e:
            if self.retry_n > self.ERR_RETRY_COUNT:
                print(f"API请求异常: {e}")
                self.retry_n = 0
                return None
            else:
                sleep(self.RETRY_DURATION)
                self.retry_n += 1
                return self.author_detail(id)
        finally:
            sleep(self.LIMIT_DELAY)  # 尊重API限制

# -------------------


class AuthorRelated:
    """
    作者相关性类：基于合作网络 BFS 距离
    """

    def __init__(self) -> None:
        self._api = S2Api()

    def set_paper(self, paper: Dict) -> None:
        self._paper: Dict = paper

    def get_paper(self) -> Dict:
        return self._paper

    def get_coauthors_by_authorid(self, author_id: str) -> Set[str]:
        """
        根据 authorId 获取其所有合作者（带缓存）
        """
        papers = self._api.author_papers(author_id)
        coauthors: Set[str] = set()

        for paper in papers:
            for author in paper.get("authors", []):
                aid = author.get("authorId")
                if aid and aid != author_id:
                    coauthors.add(aid)
        return coauthors

    def build_local_coauthor_graph(
        self, seed_authors: Set[str], max_depth: int = 2
    ) -> Dict[str, Set[str]]:
        """
        从种子作者构建局部合作图
        """
        graph: Dict[str, Set[str]] = {}
        visited: Set[str] = set(seed_authors)
        queue = deque([(a, 0) for a in seed_authors])

        while queue:
            current, depth = queue.popleft()

            if depth >= max_depth:
                continue

            neighbors = self.get_coauthors_by_authorid(current)
            graph[current] = neighbors

            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, depth + 1))

        return graph

    def min_coauthor_distance(
        self,
        citing_authors: Set[str],
        cited_authors: Set[str],
        graph: Dict[str, Set[str]],
        max_depth: int = 3,
    ) -> int:
        """
        BFS 计算最短作者距离
        """
        queue = deque()
        visited: Set[str] = set()

        for a in citing_authors:
            queue.append((a, 0))
            visited.add(a)

        while queue:
            current, depth = queue.popleft()

            if depth > max_depth:
                break

            if current in cited_authors:
                return depth

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return max_depth + 1

    def _extract_author_ids_from_paper(self, paper: Dict) -> Set[str]:
        """
        从论文 JSON 中提取 authorId 集合
        """
        return {
            a.get("authorId") for a in paper.get("authors", []) if a.get("authorId")
        }

    def compute_author_distance_score(
        self, citing_paper: Dict, cited_paper: Dict = None, max_depth: int = 3
    ) -> float:
        """
        计算 citing_paper 与初始化时 cited paper 的作者距离
        """
        if not cited_paper:
            cited_paper = self._paper
        citing_authors = self._extract_author_ids_from_paper(citing_paper)
        cited_authors = self._extract_author_ids_from_paper(cited_paper)

        if not citing_authors or not cited_authors:
            return max_depth + 1

        citing_authors_list = list(citing_authors)
        cited_authors_list = list(cited_authors)
        if citing_authors_list[0] == cited_authors_list[0]:
            return 0
        if citing_authors_list[-1] == cited_authors_list[-1]:
            return 0.1

        seed_authors = citing_authors | cited_authors

        graph = self.build_local_coauthor_graph(seed_authors, max_depth=2)

        distance = self.min_coauthor_distance(
            citing_authors, cited_authors, graph, max_depth=max_depth
        )
        if distance == 0:
            return 0.5
        elif distance == 1:
            return 0.6
        elif distance == 2:
            return 0.8
        else:
            return 1.0


# -------------------
class CitationRelevance:
    """
    Citation relevance scorer.

    Baselines:
    - MiniLM cosine: general dense semantic similarity
    - SPECTER cosine: scientific-text dense embedding similarity
    - SparseEncoder / SPLADE: sparse neural retrieval relevance
    - Cross-Encoder: pairwise neural relevance
    - SciFact-style NLI: evidence-support relevance

    Input:
        citation context(s) + cited paper abstract

    If context is a list, each context is scored independently and then
    aggregated by max or mean. Default is max.
    """

    def __init__(
        self,
        minilm_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        specter_model: Optional[str] = "sentence-transformers/allenai-specter",
        sparse_model: Optional[str] = "naver/splade-cocondenser-ensembledistil",
        cross_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scifact_model: Optional[str] = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_length: int = 512,
        default_context_agg: str = "max",
        normalize_cosine: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.max_length = max_length
        self.default_context_agg = default_context_agg
        self.normalize_cosine = normalize_cosine

        self._minilm = None
        self._specter = None
        self._sparse = None
        self._cross = None
        self._scifact_tokenizer = None
        self._scifact_model = None

        # SparseEncoder 的 dot-product 不是天然 [0, 1]，
        # 所以后面可以用 fit_sparse_normalizer 估计一个全局归一化因子。
        self._sparse_norm: Optional[float] = None

        if minilm_model:
            self._minilm = SentenceTransformer(
                minilm_model,
                device=self.device,
            )

        if specter_model:
            self._specter = SentenceTransformer(
                specter_model,
                device=self.device,
            )

        if sparse_model:
            if SparseEncoder is None:
                raise ImportError(
                    "SparseEncoder is not available. "
                    "Please upgrade sentence-transformers: "
                    "pip install -U sentence-transformers"
                )

            self._sparse = SparseEncoder(
                sparse_model,
                device=self.device,
            )

        if cross_model:
            self._cross = CrossEncoder(
                cross_model,
                device=self.device,
            )

        if scifact_model:
            self._scifact_tokenizer = AutoTokenizer.from_pretrained(
                scifact_model,
                token=self.hf_token,
            )

            self._scifact_model = AutoModelForSequenceClassification.from_pretrained(
                scifact_model,
                token=self.hf_token,
            ).to(self.device)

            self._scifact_model.eval()

    # ------------------------------------------------------------------
    # Public scoring APIs
    # ------------------------------------------------------------------

    def score_all(
        self,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Score one citation instance with all enabled methods.
        """
        return {
            "minilm": self.score_minilm(context, abstract, agg=agg)
            if self._minilm is not None else None,

            "specter": self.score_specter(context, abstract, agg=agg)
            if self._specter is not None else None,

            "sparse": self.score_sparse(context, abstract, agg=agg)
            if self._sparse is not None else None,

            "cross": self.score_cross(context, abstract, agg=agg)
            if self._cross is not None else None,

            "scifact": self.score_scifact(context, abstract, agg=agg)
            if self._scifact_model is not None else None,
        }

    def score_minilm(
        self,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
    ) -> float:
        """
        General dense embedding cosine baseline.

        Model:
            sentence-transformers/all-MiniLM-L6-v2

        Meaning:
            general semantic similarity between citation context and cited abstract.
        """
        if self._minilm is None:
            raise ValueError("MiniLM model is not loaded.")

        return self._score_dense_cosine(
            model=self._minilm,
            context=context,
            abstract=abstract,
            agg=agg,
        )

    def score_specter(
        self,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
    ) -> float:
        """
        Scientific dense embedding cosine baseline.

        Model:
            sentence-transformers/allenai-specter

        Meaning:
            scientific-text semantic similarity between citation context and cited abstract.
        """
        if self._specter is None:
            raise ValueError("SPECTER model is not loaded.")

        return self._score_dense_cosine(
            model=self._specter,
            context=context,
            abstract=abstract,
            agg=agg,
        )

    def score_sparse(
        self,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
        normalize: bool = True,
    ) -> float:
        """
        Sparse neural retrieval baseline.

        Model:
            naver/splade-cocondenser-ensembledistil

        Meaning:
            sparse lexical-semantic relevance between citation context and cited abstract.

        Raw SparseEncoder score is usually dot-product similarity, not naturally [0, 1].
        If fit_sparse_normalizer(...) has been called, global log-percentile normalization
        is used. Otherwise, a simple positive squash is used.
        """
        if self._sparse is None:
            raise ValueError("SparseEncoder model is not loaded.")

        contexts = self._as_context_list(context)
        abstract = self._clean_text(abstract)
        agg = agg or self.default_context_agg

        if not contexts or not abstract:
            return 0.0

        raw_scores = []

        for c in contexts:
            raw = self._sparse_raw_score(c, abstract)
            raw_scores.append(raw)

        raw_score = self._aggregate_scores(raw_scores, agg)

        if not normalize:
            return float(raw_score)

        if self._sparse_norm is not None and self._sparse_norm > 0:
            score = math.log1p(raw_score) / math.log1p(self._sparse_norm)
            return self._clip01(score)

        return self._positive_squash(raw_score)

    def score_cross(
        self,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
    ) -> float:
        """
        Pairwise neural relevance baseline.

        Model:
            cross-encoder/ms-marco-MiniLM-L-6-v2

        Meaning:
            neural interaction-based relevance between citation context and cited abstract.

        The MS MARCO cross-encoder returns raw logits, so sigmoid is applied to
        map scores into [0, 1].
        """
        if self._cross is None:
            raise ValueError("CrossEncoder model is not loaded.")

        contexts = self._as_context_list(context)
        abstract = self._clean_text(abstract)
        agg = agg or self.default_context_agg

        if not contexts or not abstract:
            return 0.0

        pairs = [(c, abstract) for c in contexts]
        raw_scores = self._cross.predict(pairs)

        scores = [
            self._sigmoid(float(s))
            for s in np.asarray(raw_scores).reshape(-1)
        ]

        return self._aggregate_scores(scores, agg)

    def score_scifact(
        self,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
    ) -> float:
        """
        SciFact-style NLI / evidence-support baseline.

        Input:
            premise/evidence: cited abstract
            hypothesis/claim: citation context

        Meaning:
            whether the cited abstract supports the scientific statement
            expressed in the citation context.

        Score:
            P(entailment/support | cited abstract, citation context)
        """
        if self._scifact_model is None or self._scifact_tokenizer is None:
            raise ValueError("SciFact/NLI model is not loaded.")

        contexts = self._as_context_list(context)
        abstract = self._clean_text(abstract)
        agg = agg or self.default_context_agg

        if not contexts or not abstract:
            return 0.0

        scores = []

        for c in contexts:
            score = self._nli_support_score(
                premise=abstract,
                hypothesis=c,
            )
            scores.append(score)

        return self._aggregate_scores(scores, agg)

    # ------------------------------------------------------------------
    # Optional normalizer for SparseEncoder
    # ------------------------------------------------------------------

    def fit_sparse_normalizer(
        self,
        citation_pairs: Sequence[Tuple[ContextInput, str]],
        percentile: float = 99.0,
    ) -> None:
        """
        Fit a global normalization factor for SparseEncoder scores.

        Use all citation pairs with context in the experiment.

        Example:
            citation_pairs = list(zip(df["contexts"], df["cited_abstract"]))
            relevance.fit_sparse_normalizer(citation_pairs, percentile=99.0)
        """
        if self._sparse is None:
            raise ValueError("SparseEncoder model is not loaded.")

        raw_scores = []

        for context, abstract in citation_pairs:
            contexts = self._as_context_list(context)
            abstract = self._clean_text(abstract)

            if not contexts or not abstract:
                continue

            for c in contexts:
                raw = self._sparse_raw_score(c, abstract)
                if raw > 0:
                    raw_scores.append(raw)

        if not raw_scores:
            self._sparse_norm = 1.0
            return

        self._sparse_norm = float(np.percentile(raw_scores, percentile))

        if self._sparse_norm <= 0:
            self._sparse_norm = float(max(raw_scores))

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _score_dense_cosine(
        self,
        model: SentenceTransformer,
        context: ContextInput,
        abstract: str,
        agg: Optional[str] = None,
    ) -> float:
        contexts = self._as_context_list(context)
        abstract = self._clean_text(abstract)
        agg = agg or self.default_context_agg

        if not contexts or not abstract:
            return 0.0

        texts = contexts + [abstract]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        context_embeddings = embeddings[:-1]
        abstract_embedding = embeddings[-1]

        scores = []

        for e in context_embeddings:
            cos = float(np.dot(e, abstract_embedding))

            if self.normalize_cosine:
                score = (cos + 1.0) / 2.0
            else:
                score = cos

            scores.append(self._clip01(score))

        return self._aggregate_scores(scores, agg)

    def _sparse_raw_score(
        self,
        context: str,
        abstract: str,
    ) -> float:
        """
        Compute raw SparseEncoder similarity score.
        """
        context = self._clean_text(context)
        abstract = self._clean_text(abstract)

        if not context or not abstract:
            return 0.0

        embeddings = self._sparse.encode(
            [context, abstract],
            show_progress_bar=False,
        )

        sim = self._sparse.similarity(
            embeddings[0],
            embeddings[1],
        )

        # Different sentence-transformers versions may return tensor / ndarray / matrix.
        if hasattr(sim, "detach"):
            sim = sim.detach().cpu().numpy()

        sim = np.asarray(sim).reshape(-1)

        if len(sim) == 0:
            return 0.0

        return float(max(0.0, sim[0]))

    def _nli_support_score(
        self,
        premise: str,
        hypothesis: str,
    ) -> float:
        """
        premise:
            cited abstract

        hypothesis:
            citation context

        Return:
            probability of entailment/support.
        """
        premise = self._clean_text(premise)
        hypothesis = self._clean_text(hypothesis)

        if not premise or not hypothesis:
            return 0.0

        inputs = self._scifact_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self._scifact_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        probs_np = probs.detach().cpu().numpy()

        id2label = self._scifact_model.config.id2label

        entailment_idx = self._find_label_index(
            id2label=id2label,
            keywords=[
                "entailment",
                "entail",
                "support",
                "supports",
                "supported",
            ],
        )

        if entailment_idx is None:
            # 常见 NLI 顺序：contradiction, neutral, entailment
            # 如果模型没有清楚的 label 名，就保守使用最后一类作为 entailment。
            entailment_idx = len(probs_np) - 1

        return self._clip01(float(probs_np[entailment_idx]))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: Any) -> str:
        if text is None:
            return ""

        if isinstance(text, float) and np.isnan(text):
            return ""

        text = str(text)
        text = " ".join(text.split())

        return text.strip()

    def _as_context_list(
        self,
        context: ContextInput,
    ) -> List[str]:
        if context is None:
            return []

        if isinstance(context, str):
            text = self._clean_text(context)
            return [text] if text else []

        contexts = []

        for c in context:
            text = self._clean_text(c)
            if text:
                contexts.append(text)

        return contexts

    def _aggregate_scores(
        self,
        scores: Sequence[float],
        agg: Optional[str] = None,
    ) -> float:
        if not scores:
            return 0.0

        agg = agg or self.default_context_agg

        scores = [
            float(s)
            for s in scores
            if s is not None and not np.isnan(float(s))
        ]

        if not scores:
            return 0.0

        if agg == "max":
            return float(np.max(scores))

        if agg == "mean":
            return float(np.mean(scores))

        if agg == "median":
            return float(np.median(scores))

        raise ValueError(f"Unknown aggregation method: {agg}")

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)

        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _positive_squash(x: float) -> float:
        x = max(0.0, float(x))
        return x / (1.0 + x)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _find_label_index(
        id2label: Dict[int, str],
        keywords: Sequence[str],
    ) -> Optional[int]:
        for idx, label in id2label.items():
            label_lower = str(label).lower()

            for kw in keywords:
                if kw in label_lower:
                    return int(idx)

        return None

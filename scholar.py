import ast
import json
import os
from collections import deque
from time import sleep, time
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

load_dotenv()


class S2Api:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    API_KEY = os.getenv("API_KEY")
    RETRY_DURATION = 2
    LIMIT_DELAY = 2
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
                # print(f"已清理 {cache_type} 缓存")
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

    def paper_detail(self, id: str, use_cache: bool = True) -> Optional[Dict]:
        # 检查缓存
        if use_cache:
            cached_data = self._get_from_cache("paper_detail", id)
            if cached_data is not None:
                return cached_data

        url = f"{self.BASE_URL}/paper/{id}"
        params = {
            "fields": "paperId,paperId,title,abstract,authors.name,authors.affiliations,year,publicationDate,venue,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,url,openAccessPdf,externalIds,corpusId,publicationDate,isOpenAccess,publicationTypes,journal,citationStyles,embedding.specter_v2,tldr,citations.title,citations.abstract"
        }
        try:
            resp = requests.get(url, params=params, headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                data = resp.json()
                if use_cache:
                    self._save_to_cache("paper_detail", id, data)
                    # print(f"[缓存保存] 论文详情: {id}")
                return data
            elif resp.status_code == 429:
                print("速率限制，等待后重试...")
                sleep(self.RETRY_DURATION)
                return self.paper_detail(id)
            else:
                print(f"API请求失败: {resp.status_code} - {resp.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"API请求异常: {e}")
            return None
        finally:
            sleep(self.LIMIT_DELAY)  # 尊重API限制

    def paper_citations(self, id, use_cache: bool = True) -> Optional[List[Dict]]:
        if use_cache:
            cached_data = self._get_from_cache("paper_citations", id)
            if cached_data is not None:
                return cached_data

        url = f"{self.BASE_URL}/paper/{id}/citations"
        params = {"fields": "title,abstract,contexts,intents,isInfluential,authors"}
        try:
            resp = requests.get(url, params=params, headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                data = resp.json()["data"] or []
                if use_cache:
                    self._save_to_cache("paper_citations", id, data)
                    # print(f"[缓存保存] 论文引用: {id}")
                return data
            elif resp.status_code == 429:
                print("速率限制，等待后重试...")
                sleep(self.RETRY_DURATION)
                return self.paper_citations(id)
            else:
                print(f"API请求失败: {resp.status_code} - {resp.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"API请求异常: {e}")
            return None
        finally:
            sleep(self.LIMIT_DELAY)  # 尊重API限制

    def author_papers(self, id: str, use_cache: bool = True) -> Optional[List[Dict]]:
        if use_cache:
            cached_data = self._get_from_cache("author_papers", id)
            if cached_data is not None:
                return cached_data

        url = f"{self.BASE_URL}/author/{id}/papers"
        params = {"fields": "title,year,publicationDate"}
        try:
            resp = requests.get(url, params=params, headers={"x-api-key": self.API_KEY})
            if resp.status_code == 200:
                data = resp.json()["data"] or []
                if use_cache:
                    self._save_to_cache("author_papers", id, data)
                    # print(f"[缓存保存] 作者论文: {id}")
                return data
            elif resp.status_code == 429:
                print("速率限制，等待后重试...")
                sleep(self.RETRY_DURATION)
                return self.paper_citations(id)
            else:
                print(f"API请求失败: {resp.status_code} - {resp.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"API请求异常: {e}")
            return None
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
    Citation relevance scorer with two modes:
    - cosine: SPECTER embedding cosine similarity
    - cross:  Cross-Encoder semantic relevance
    Output score in [0,1].
    """

    def __init__(
        self,
        specter_model: str = "allenai-specter",
        cross_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._specter = SentenceTransformer(specter_model)
        self._cross = CrossEncoder(cross_model)
        self._paper: Optional[Dict] = None

    def set_paper(self, paper: Dict) -> None:
        """Set cited paper JSON."""
        self._paper = paper

    def get_paper(self) -> Optional[Dict]:
        return self._paper

    # ---------- utils ----------

    @staticmethod
    def _sigmoid(x: float, T=2.0) -> float:
        # return 1.0 / (1.0 + np.exp(-x))
        return 1.0 / (1.0 + np.exp(-x / T))

    @staticmethod
    def _normalize_cosine(cos: float) -> float:
        """Map cosine similarity from [-1,1] to [0,1]."""
        return (cos + 1.0) / 2.0

    @staticmethod
    def _parse_context(contexts):
        if isinstance(contexts, str):
            try:
                contexts = ast.literal_eval(contexts)
            except Exception:
                contexts = [contexts]
        if isinstance(contexts, list):
            return [c for c in contexts if isinstance(c, str) and len(c.strip()) > 0]
        return []

    # ---------- core scorers ----------

    def _cross_score(self, texts: List[str], abstract: str) -> float:
        pairs = [(t, abstract) for t in texts if t.strip()]
        if not pairs:
            return 0.0
        scores = self._cross.predict(pairs)
        return float(np.max(scores))

    def _cosine_score(self, texts: List[str], abstract: str) -> float:
        v_abs = self._specter.encode(abstract)
        v_txts = self._specter.encode(texts)
        sims = cosine_similarity([v_abs], v_txts)[0]
        return float(np.max(sims))

    # ---------- public API ----------

    def compute_relevance(
        self, citing: List[str] | str, abstract: str = None, mode: str = "cross"
    ) -> float:
        """
        mode: 'cosine' or 'cross'
        Return relevance score in [0,1].
        """

        if abstract is None:
            if self._paper:
                abstract = self._paper.get("abstract")
                if not abstract:
                    raise ValueError("Cited paper not set.")
            else:
                raise ValueError("Cited paper not set.")

        contexts = self._parse_context(citing)

        if len(contexts) == 0:
            return 0.0

        if mode == "cosine":
            cos_raw = self._cosine_score(contexts, abstract)
            return float(np.clip(self._normalize_cosine(cos_raw), 0.0, 1.0))

        if mode == "cross":
            cross_raw = self._cross_score(contexts, abstract)
            return float(np.clip(self._sigmoid(cross_raw), 0.0, 1.0))

        raise ValueError(f"Unknown mode: {mode}")

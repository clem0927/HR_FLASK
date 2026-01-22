# src/gyu/models/base_recommender.py
"""
포상 추천 엔진 추상 베이스 클래스
- TF-IDF, KoBERT 등 다양한 구현체가 이 인터페이스를 따름
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRecommender(ABC):
    """포상 추천 엔진의 추상 베이스 클래스"""

    @abstractmethod
    def analyze(self, comments: List[str], policies: List[Dict]) -> Dict[str, Any]:
        """
        평가 코멘트를 분석하여 포상 정책과 매칭

        Args:
            comments: 평가 코멘트 리스트
            policies: 포상 정책 리스트 [{"policyId": 1, "policyName": "...", "description": "..."}]

        Returns:
            {
                "recommendations": [...],
                "extractedKeywords": [...],
                "overallSentiment": "positive" | "negative" | "neutral"
            }
        """
        pass

    @abstractmethod
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        텍스트에서 키워드 추출

        Args:
            text: 분석할 텍스트
            top_n: 추출할 키워드 수

        Returns:
            키워드 리스트
        """
        pass

    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 유사도 계산

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        pass

    def get_model_type(self) -> str:
        """모델 타입 반환"""
        return self.__class__.__name__
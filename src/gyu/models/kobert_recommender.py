# src/gyu/models/kobert_recommender.py
"""
PyTorch + Sentence-BERT 기반 포상 추천 엔진
- 한국어 문장 임베딩 모델 사용
- 코사인 유사도로 정책 매칭
"""
import torch
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, util

from .base_recommender import BaseRecommender
from ..services.text_preprocessor import TextPreprocessor
from ..config import (
    RECOMMENDATION_CONFIG,
    SENTIMENT_KEYWORDS,
    POLICY_KEYWORDS,
    KOBERT_CONFIG
)


class KoBertRecommender(BaseRecommender):
    """PyTorch 기반 한국어 Sentence-BERT 추천 엔진"""

    def __init__(self):
        """모델 초기화"""
        self.preprocessor = TextPreprocessor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 한국어 Sentence-BERT 모델 로드
        # 'jhgan/ko-sroberta-multitask': 한국어 문장 유사도에 최적화된 모델
        model_name = "jhgan/ko-sroberta-multitask"
        self.model = SentenceTransformer(model_name, device=self.device)

        print(f"[KoBertRecommender] 모델 로드 완료: {model_name}")
        print(f"[KoBertRecommender] 디바이스: {self.device}")

    def analyze(self, comments: List[str], policies: List[Dict]) -> Dict[str, Any]:
        """
        평가 코멘트를 분석하여 포상 정책과 매칭

        Args:
            comments: 평가 코멘트 리스트
            policies: 포상 정책 리스트

        Returns:
            분석 결과
        """
        if not comments or not policies:
            return {
                "recommendations": [],
                "extractedKeywords": [],
                "overallSentiment": "neutral"
            }

        # 코멘트 결합
        combined_comments = self.preprocessor.join_texts(comments)

        # 키워드 추출
        extracted_keywords = self.extract_keywords(combined_comments)

        # 감정 분석
        sentiment = self.preprocessor.analyze_sentiment(
            combined_comments,
            SENTIMENT_KEYWORDS["positive"],
            SENTIMENT_KEYWORDS["negative"]
        )

        # 부정적 평가는 포상 추천 제외
        if sentiment == "negative":
            return {
                "recommendations": [],
                "extractedKeywords": extracted_keywords,
                "overallSentiment": sentiment,
                "skipReason": "부정적 평가가 많아 포상 추천 대상에서 제외되었습니다."
            }

        # 코멘트 임베딩 생성 (한 번만 계산)
        comment_embedding = self.model.encode(combined_comments, convert_to_tensor=True)

        # 각 정책에 대한 매칭 점수 계산
        recommendations = []
        for policy in policies:
            recommendation = self._match_policy(
                combined_comments,
                comment_embedding,
                policy,
                extracted_keywords
            )
            if recommendation and recommendation["matchScore"] >= RECOMMENDATION_CONFIG["min_match_score"]:
                recommendations.append(recommendation)

        # 점수순 정렬
        recommendations.sort(key=lambda x: x["matchScore"], reverse=True)

        # 최대 추천 수 제한
        max_recs = RECOMMENDATION_CONFIG["max_recommendations"]
        recommendations = recommendations[:max_recs]

        return {
            "recommendations": recommendations,
            "extractedKeywords": extracted_keywords,
            "overallSentiment": sentiment
        }

    def _match_policy(
        self,
        comments_text: str,
        comment_embedding: torch.Tensor,
        policy: Dict,
        extracted_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        개별 정책과 코멘트 매칭 (BERT 임베딩 기반)

        Args:
            comments_text: 결합된 코멘트 텍스트
            comment_embedding: 코멘트의 BERT 임베딩
            policy: 포상 정책
            extracted_keywords: 추출된 키워드

        Returns:
            매칭 결과
        """
        policy_name = policy.get("policyName", "")
        policy_description = policy.get("description", policy_name)

        # 정책 관련 키워드 가져오기
        policy_keywords = POLICY_KEYWORDS.get(policy_name, [])

        # 방법 1: BERT 임베딩 코사인 유사도
        policy_text = f"{policy_name}. {policy_description}"
        policy_embedding = self.model.encode(policy_text, convert_to_tensor=True)
        similarity_score = util.cos_sim(comment_embedding, policy_embedding).item()

        # 방법 2: 키워드 매칭 (보조)
        matched_keywords = []
        keyword_score = 0
        if policy_keywords:
            for keyword in policy_keywords:
                if keyword in comments_text:
                    matched_keywords.append(keyword)
            keyword_score = len(matched_keywords) / len(policy_keywords) if policy_keywords else 0

        # 최종 점수 계산 (BERT 유사도 70% + 키워드 매칭 30%)
        # BERT 유사도는 -1~1 범위이므로 0~1로 정규화
        normalized_similarity = (similarity_score + 1) / 2
        final_score = (normalized_similarity * 0.7 + keyword_score * 0.3) * 100
        final_score = min(100, int(final_score))

        # 매칭 사유 생성
        reason = self._generate_reason(policy_name, matched_keywords, similarity_score)

        return {
            "policyId": policy.get("policyId"),
            "policyName": policy_name,
            "matchScore": final_score,
            "similarityScore": round(similarity_score, 4),
            "reason": reason,
            "keywords": matched_keywords
        }

    def _generate_reason(self, policy_name: str, matched_keywords: List[str], similarity: float) -> str:
        """매칭 사유 생성"""
        reasons = []

        if similarity >= 0.5:
            reasons.append(f"'{policy_name}' 정책과 높은 의미적 연관성 확인 (BERT 분석)")
        elif similarity >= 0.3:
            reasons.append(f"'{policy_name}' 정책과 연관성 확인")
        elif similarity >= 0.1:
            reasons.append(f"'{policy_name}' 정책과 약한 연관성")

        if matched_keywords:
            keyword_str = "', '".join(matched_keywords[:3])
            reasons.append(f"핵심 키워드 '{keyword_str}' 발견")

        if not reasons:
            reasons.append("기본 조건 충족")

        return ". ".join(reasons) + "."

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        텍스트에서 키워드 추출 (형태소 분석 기반)

        Args:
            text: 분석할 텍스트
            top_n: 추출할 키워드 수

        Returns:
            키워드 리스트
        """
        if not text:
            return []

        # 형태소 분석으로 명사 추출
        tokens = self.preprocessor.tokenize(text)
        if not tokens:
            return []

        # 빈도 기반 상위 키워드
        from collections import Counter
        word_freq = Counter(tokens)
        top_words = [word for word, _ in word_freq.most_common(top_n)]

        return top_words

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 코사인 유사도 계산 (BERT 임베딩 기반)

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            유사도 점수 (-1.0 ~ 1.0, 보통 0.0 ~ 1.0)
        """
        if not text1 or not text2:
            return 0.0

        # BERT 임베딩 생성
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)

        # 코사인 유사도 계산
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        return float(similarity)
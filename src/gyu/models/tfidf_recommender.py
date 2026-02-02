# src/gyu/models/tfidf_recommender.py
"""
TF-IDF 기반 포상 추천 엔진
- scikit-learn의 TfidfVectorizer 사용
- 코사인 유사도로 정책 매칭
"""
from typing import List, Dict, Any
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_recommender import BaseRecommender
from ..services.text_preprocessor import TextPreprocessor
from ..config import (
    TFIDF_CONFIG,
    RECOMMENDATION_CONFIG,
    SENTIMENT_KEYWORDS,
    POLICY_KEYWORDS
)


class TfidfRecommender(BaseRecommender):
    """TF-IDF 기반 포상 추천 엔진"""

    def __init__(self):
        """TF-IDF 추천 엔진 초기화"""
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_CONFIG["max_features"],
            min_df=TFIDF_CONFIG["min_df"],
            max_df=TFIDF_CONFIG["max_df"],
            ngram_range=TFIDF_CONFIG["ngram_range"],
            tokenizer=self._custom_tokenizer,
            token_pattern=None  # 커스텀 토크나이저 사용 시 필요
        )
        print("[TfidfRecommender] 초기화 완료")

    def _custom_tokenizer(self, text: str) -> List[str]:
        """커스텀 토크나이저 (형태소 분석 사용)"""
        return self.preprocessor.tokenize(text)

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

        # 각 정책에 대한 매칭 점수 계산
        recommendations = []
        for policy in policies:
            recommendation = self._match_policy(combined_comments, policy, extracted_keywords)
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

    def _match_policy(self, comments_text: str, policy: Dict, extracted_keywords: List[str]) -> Dict[str, Any]:
        """
        개별 정책과 코멘트 매칭

        Args:
            comments_text: 결합된 코멘트 텍스트
            policy: 포상 정책
            extracted_keywords: 추출된 키워드

        Returns:
            매칭 결과
        """
        policy_name = policy.get("policyName", "")
        policy_description = policy.get("description", policy_name)

        # 정책 관련 키워드 가져오기
        policy_keywords = POLICY_KEYWORDS.get(policy_name, [])

        # 방법 1: TF-IDF 유사도 계산
        similarity_score = self.calculate_similarity(comments_text, policy_description)

        # 방법 2: 키워드 매칭
        matched_keywords = []
        keyword_score = 0
        if policy_keywords:
            for keyword in policy_keywords:
                if keyword in comments_text:
                    matched_keywords.append(keyword)
            keyword_score = len(matched_keywords) / len(policy_keywords) if policy_keywords else 0

        # 최종 점수 계산 (유사도 + 키워드 매칭)
        final_score = (similarity_score * 0.5 + keyword_score * 0.5) * 100
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
            reasons.append(f"'{policy_name}' 정책과 높은 연관성 확인")
        elif similarity >= 0.3:
            reasons.append(f"'{policy_name}' 정책과 연관성 확인")

        if matched_keywords:
            keyword_str = "', '".join(matched_keywords[:3])
            reasons.append(f"핵심 키워드 '{keyword_str}' 발견")

        if not reasons:
            reasons.append("기본 조건 충족")

        return ". ".join(reasons) + "."

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        TF-IDF를 사용한 키워드 추출

        Args:
            text: 분석할 텍스트
            top_n: 추출할 키워드 수

        Returns:
            키워드 리스트
        """
        if not text:
            return []

        # 토큰화
        tokens = self.preprocessor.tokenize(text)
        if not tokens:
            return []

        # 문서가 하나뿐이므로 단순 빈도 기반으로 추출
        from collections import Counter
        word_freq = Counter(tokens)
        top_words = [word for word, _ in word_freq.most_common(top_n)]

        return top_words

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 코사인 유사도 계산

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        if not text1 or not text2:
            return 0.0

        try:
            # TF-IDF 벡터화
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])

            # 코사인 유사도 계산
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            return float(similarity)
        except Exception as e:
            print(f"[TfidfRecommender] 유사도 계산 오류: {e}")
            return 0.0


# 팩토리 함수
def get_recommender() -> BaseRecommender:
    """
    설정에 따른 추천 엔진 반환

    Returns:
        BaseRecommender 구현체
    """
    from ..config import RECOMMENDER_TYPE

    if RECOMMENDER_TYPE == "kobert":
        # KoBERT 구현체가 있을 경우
        try:
            from .kobert_recommender import KoBertRecommender
            return KoBertRecommender()
        except ImportError:
            print("[WARNING] KoBERT 미설치, TF-IDF로 대체")
            return TfidfRecommender()
    else:
        return TfidfRecommender()
# src/gyu/services/text_preprocessor.py
"""
한글 텍스트 전처리 모듈
- 형태소 분석, 불용어 제거, 정규화 등
"""
import re
from typing import List, Optional

# kiwipiepy 사용 (설치 필요: pip install kiwipiepy)
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False


class TextPreprocessor:
    """한글 텍스트 전처리 클래스"""

    # 한글 불용어 리스트
    STOPWORDS = {
        '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
        '도', '를', '으로', '자', '에', '와', '한', '하다', '것', '수',
        '있다', '없다', '되다', '이다', '그', '저', '등', '및', '더',
        '에서', '로', '에게', '까지', '부터', '만', '또', '또한'
    }

    # 추출할 품사 태그 (명사, 동사, 형용사)
    VALID_POS = {'NNG', 'NNP', 'VV', 'VA', 'XR'}

    def __init__(self):
        """전처리기 초기화"""
        self.kiwi = None
        if KIWI_AVAILABLE:
            self.kiwi = Kiwi()
            print("[TextPreprocessor] Kiwi 형태소 분석기 초기화 완료")
        else:
            print("[TextPreprocessor] Kiwi 미설치 - 기본 토크나이저 사용")

    def preprocess(self, text: str) -> str:
        """
        텍스트 전처리 (정규화)

        Args:
            text: 원본 텍스트

        Returns:
            전처리된 텍스트
        """
        if not text:
            return ""

        # 소문자 변환 (영문)
        text = text.lower()

        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)

        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def tokenize(self, text: str, use_pos_filter: bool = True) -> List[str]:
        """
        텍스트 토큰화 (형태소 분석)

        Args:
            text: 분석할 텍스트
            use_pos_filter: 품사 필터링 여부

        Returns:
            토큰 리스트
        """
        if not text:
            return []

        text = self.preprocess(text)

        if self.kiwi:
            return self._tokenize_with_kiwi(text, use_pos_filter)
        else:
            return self._tokenize_simple(text)

    def _tokenize_with_kiwi(self, text: str, use_pos_filter: bool) -> List[str]:
        """Kiwi를 사용한 형태소 분석"""
        tokens = []
        result = self.kiwi.tokenize(text)

        for token in result:
            # 품사 필터링
            if use_pos_filter and token.tag not in self.VALID_POS:
                continue

            # 불용어 제거
            if token.form in self.STOPWORDS:
                continue

            # 1글자 제거 (의미 없는 토큰)
            if len(token.form) < 2:
                continue

            tokens.append(token.form)

        return tokens

    def _tokenize_simple(self, text: str) -> List[str]:
        """간단한 공백 기반 토큰화 (Kiwi 미설치 시)"""
        tokens = text.split()
        return [
            token for token in tokens
            if token not in self.STOPWORDS and len(token) >= 2
        ]

    def extract_nouns(self, text: str) -> List[str]:
        """
        텍스트에서 명사만 추출

        Args:
            text: 분석할 텍스트

        Returns:
            명사 리스트
        """
        if not text:
            return []

        text = self.preprocess(text)

        if self.kiwi:
            nouns = []
            result = self.kiwi.tokenize(text)
            for token in result:
                if token.tag in {'NNG', 'NNP'} and len(token.form) >= 2:
                    if token.form not in self.STOPWORDS:
                        nouns.append(token.form)
            return nouns
        else:
            return self._tokenize_simple(text)

    def join_texts(self, texts: List[str]) -> str:
        """
        여러 텍스트를 하나로 결합

        Args:
            texts: 텍스트 리스트

        Returns:
            결합된 텍스트
        """
        return ' '.join(filter(None, texts))

    def analyze_sentiment(self, text: str, positive_words: List[str], negative_words: List[str]) -> str:
        """
        간단한 감정 분석

        Args:
            text: 분석할 텍스트
            positive_words: 긍정 키워드 리스트
            negative_words: 부정 키워드 리스트

        Returns:
            "positive", "negative", 또는 "neutral"
        """
        if not text:
            return "neutral"

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
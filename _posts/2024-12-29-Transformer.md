# Transformer: 딥러닝의 혁신적인 아키텍처

Transformer 모델은 Vaswani et al.이 2017년에 발표한 논문 *"Attention Is All You Need"*에서 소개되었으며, 자연어 처리(NLP)와 시퀀스-투-시퀀스(Sequence-to-Sequence) 작업에 혁신을 가져왔습니다. 이 모델은 기존의 순환 신경망(RNN) 및 합성곱 신경망(CNN)을 대체하며, 긴 문맥 정보를 효율적으로 처리할 수 있는 능력으로 주목받고 있습니다.

---

## Transformer의 주요 특징

### 1. **Self-Attention 메커니즘**

- 입력 시퀀스의 특정 토큰을 처리할 때, 다른 토큰에 집중할 수 있도록 해줍니다.
- 각 토큰이 시퀀스 내에서 다른 토큰의 중요도를 계산하여 의미 있는 관계를 학습합니다.
- 계산 공식:  
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \]
  - \( Q \): Query 행렬
  - \( K \): Key 행렬
  - \( V \): Value 행렬
  - \( d_k \): Key 벡터의 차원

### 2. **Positional Encoding**

- Transformer는 RNN처럼 순서를 처리하는 구조가 없기 때문에, 입력 데이터에 순서를 반영하기 위해 Positional Encoding을 추가합니다.
- 이 값은 각 토큰의 위치 정보를 임베딩에 더해주는 방식으로 작동합니다.
- 대표적인 공식:  
  \[
  PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d*{\text{model}}}}}\right) \\
  PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d*{\text{model}}}}}\right)
  \]
  - \( pos \): 위치 값
  - \( i \): 임베딩 차원의 인덱스
  - \( d\_{\text{model}} \): 임베딩 벡터의 차원

---

## Transformer 구조

### 1. **Encoder**

- 입력 데이터를 처리하는 부분으로, 여러 층의 Self-Attention과 Feed-Forward 신경망으로 구성됩니다.
- 주요 단계:
  1.  입력 임베딩 + Positional Encoding
  2.  Multi-Head Self-Attention
  3.  Position-Wise Feed-Forward Network

### 2. **Decoder**

- 출력 데이터를 생성하는 부분으로, Encoder에서 전달된 정보와 함께 처리합니다.
- 주요 단계:
  1.  입력 임베딩 + Positional Encoding
  2.  Masked Multi-Head Self-Attention (미래 정보 차단)
  3.  Encoder-Decoder Attention
  4.  Position-Wise Feed-Forward Network

---

## Multi-Head Attention

- Self-Attention을 여러 번 병렬로 수행하여 서로 다른 의미적 정보를 학습합니다.
- 각 Head가 서로 다른 부분에 집중할 수 있도록 설계되었습니다.
- 장점: 모델이 다양한 문맥을 고려할 수 있음.

---

## Transformer의 장점

1. **병렬 처리 가능**: RNN과 달리 모든 토큰을 동시에 처리할 수 있어 학습 속도가 빠릅니다.
2. **긴 문맥 정보 처리**: Self-Attention으로 먼 거리의 의존성을 효과적으로 학습합니다.
3. **다양한 응용 가능**: NLP뿐만 아니라 이미지 처리, 음성 인식 등에도 활용됩니다.

---

## Transformer 기반 모델

- **BERT (Bidirectional Encoder Representations from Transformers)**: 양방향 문맥 이해를 위한 Encoder 기반 모델.
- **GPT (Generative Pre-trained Transformer)**: 텍스트 생성을 위한 Decoder 기반 모델.
- **T5 (Text-to-Text Transfer Transformer)**: 입력과 출력을 모두 텍스트로 처리하는 통합 모델.

---

## 참고 자료

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). _Attention Is All You Need._ NeurIPS.
- [Transformer 구현 코드](https://github.com/tensorflow/tensor2tensor)
- [BERT 설명 및 활용](https://github.com/google-research/bert)

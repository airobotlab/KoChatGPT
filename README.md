# kochatgpt 데이터 구축 코드
chatgpt의 RLHF를 학습하기 위한 3가지 step의 한국어 데이터셋
  - **data_kochatgpt/kochatgpt_1_SFT.jsonl** : Step1) SFT(지도학습) 학습 데이터셋
  - **data_kochatgpt/kochatgpt_1_SFT_conversation.jsonl** : Step1) SFT 학습 데이터셋(대화)
  - **data_kochatgpt/kochatgpt_2_RM.jsonl** : Step2) RM(보상모델) 학습 데이터셋
  - **data_kochatgpt/kochatgpt_3_PPO.jsonl** : Step3) PPO(강화학습) 학습 데이터셋
  - **data_kochatgpt/kochatgpt_seed_data.txt** : 한국어 질문 수집 데이터셋

# kochatgpt 실습코드
<a href="https://bit.ly/401rCrd">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

한국어 chatgpt 데이터셋으로 ChatGPT-replica를 만드는 실습코드
RLHF(Reinforcement Learning from Human Feedback)의 3단계
- Step1) SFT(지도학습)
- Step2) RM(보상모델)
- Step3) PPO(강화학습)




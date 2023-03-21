# KoChatGPT-replica(RLHF) 프로젝트

ChatGPT. 활용을 넘어서 ChatGPT-replica 모델을 직접 만들어 봅시다. ChatGPT는 공개 코드가 없습니다. 본 세미나에서는 ChatGPT를 만든 원리인 GPT fine-tuning, 강화학습(PPO), RLHF, ChatGPT 데이터셋 구축에 대해 다루고 코드 실습을 합니다. 만들어진 모델을 활용만 하는 건 재미없잖아요?? 우리 분야만의 ChatGPT(한국어/전문분야)를 직접 만드는 방법을 소개합니다.
  ※ 구현 모델은 ChatGPT-replica입니다. 실제 ChatGPT와 다를 수 있습니다.
  ※ GPT3가 아닌 GPT2+RLHF로 구현합니다. 거대언어모델로 개발시 어려움이 있을 수 있습니다.
  ※ 실습환경: Jupyter or Colab, 선수 지식: 파이썬

####  ChatGPT-replica 실습 Requirement
- 데이터(RLHF): [data_kochatgpt](data_kochatgpt)
- LLM 모델: GPT2([skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2))
- GPU: [Colab](https://colab.research.google.com/?hl=ko)


# 1) kochatgpt [데이터 구축코드](https://github.com/airobotlab/KoChatGPT/blob/main/kochatgpt_data_230320.ipynb)
chatgpt의 RLHF를 학습하기 위한 3가지 step의 한국어 데이터셋

#### 1.1) 한국어 질문 데이터셋에서 질문 수집
- **data_kochatgpt/kochatgpt_seed_data.txt** : 한국어 질문 수집 데이터셋 
    - 예시
```
불고기용 고기 한우에요?
쓰던 앱이 유료로 전환됐어
여친이랑 다툼
술 먹고 싶어
잊고싶다.
입냄새 안나나?
새로운 사랑은 찾아와
이명박 대통형은 어느 경축사를 통해 집권 후반기 국정운영에 대한 언급을 하였나?
금액은 얼마에요
리처드 닉슨이 43대 부통령직을 수행한 년도는?
```
- 출처
    - [data1, ChatbotData/11824문장](https://github.com/songys/Chatbot_data/blob/master/ChatbotData.csv)
    - [data2, AI허브_한국어 대화/49711문장](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=116)
    - [data3, AI허브_일반상식/100268](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=106)
    - [data4, KorQuad/5774](https://korquad.github.io/KorQuad%201.0/)  
    - 4가지 데이터셋에서 총 167,577 질문셋 수집
    - 12,000개 질문 랜덤 추출

### RLHF 데이터셋 생성
#### 1.2)  step1, SFT(Supervised Fine-tuning) 데이터셋 구축
###### 1.2.1) 한국어 QA 데이터셋의 질문으로 답변 자동 생성(ChatGPT): 12,000개
- SFT 데이터: [**data_kochatgpt/kochatgpt_1_SFT.jsonl**](https://github.com/airobotlab/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl)
- 다음 단어만 잘 생성하는 모델에서 질문에 '잘' 대답하는 모델 만들기
- Human labling이 필요하지만 ChatGPT 이용
- 질문을 입력으로 주고 ChatGPT로 답변 자동생성
- 데이터셋 예시

<img src="img/data_img_1.JPG" width="500">

```json
{
    "prompt": "술 먹고 싶어",
    "completion": "'술은 알코올이 함유된 음료수이며, 건강에 나쁜 영향을 미칠 수 있습니다. 따라서 술을 마시기 전에 본인의 건강 상태와 함께 적당한 양을 마시는 것이 중요합니다. 술을 마시기 전에 반드시 20세 이상인지 확인하시고, 규정된 양 이내에서 적절하게 즐기시길 바랍니다. 또한 술을 마시고 운전하는 것은 절대 금지하시기 바랍니다.",
    "tokens": 189
},
```

###### 1.2.2) langchain을 이용한 채팅데이터 자동생성(ChatGPT): 439개
- SFT 데이터: [**data_kochatgpt/kochatgpt_1_SFT_conversation.jsonl**](https://github.com/airobotlab/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT_conversation.jsonl) : 한국어 질문 수집 데이터셋 
- 다음 단어만 잘 생성하는 모델에서 질문에 '잘' 대답하는 모델 만들기
- ChatGPT로 사람과 AI가 대화를 주고받는 채팅 데이터 생성
- 데이터셋 예시
- ```json
<<start conversation>>
<사람>: 그렇군요. 무엇에 대해 이야기하고 싶으세요?
<챗봇>: 궁금하신 주제에 대해 이야기하고 싶으신가요? 예를 들어 여행, 식품, 스포츠, 등등 다양한 토론에 참여할 수 있습니다.
<사람>: 그래서 스포츠에 대해 이야기해볼까요?
<챗봇>: 그렇습니다. 스포츠에 대해 말씀하시겠습니까? 스포츠는 다양한 종목과 관련된 주제로 비슷한 경기를 비교하거나 새로운 경기를 분석할 수 있습니다. 또한 스포츠 관련 소식과 업데이트 등을 제공할 수 있습니다.
<사람>: 괜찮습니다. 소식이나 업데이트는 어떻게 받을 수 있나요?
<챗봇>: 귀하는 스포츠 관련 소식과 업데이트를 접할 수 있는 다양한 방법을 사용할 수 있습니다. 예를 들면 스포츠 사이트, 스포츠 앱, 스포츠 유튜브 채널 등을 통해 스포츠 소식과 업데이트를 받을 수 있습니다. 또한 소셜 미디어에서도 스포츠 소식과 업데이트를 받을 수 있습니다.
<사람>: 감사합니다. 이제 스포츠 소식과 업데이트를 받아볼 준비가 되었습니다.
<챗봇>: 그렇게 되었군요. 스포츠 소식과 업데이트를 받기 위해 귀하가 해야 할 준비는 무엇인가요? 스포츠 사이트, 스포츠 앱, 스포츠 유튜브 채널, 소셜 미디어 등을 통해 스포츠 소식과 업데이트를 받기 위해 준비해야 합니다.
```


#### 1.3)  step2, RM(Reward Model, 좋은글 채점모델) 데이터셋 구축
- RM 데이터: [**data_kochatgpt/kochatgpt_2_RM.jsonl**](https://github.com/airobotlab/KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl): 10,220개
- Ranking 데이터가 필요하므로 동일한 prompt에 대해 각기 다른 3가지 답변 자동 생성
    - 1) ChatGPT로 생성
    - 2) GPT3로 생성('text-davinci-003')
    - 3) GPT3로 생성('text-ada-001')
- ChatGPT > GPT3-davinci > GPT3-ada 순으로 랜덤하게 섞은 후 ranking 자동 생성
- 데이터셋 예시
```json
[
    {
        "prompt": "번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?",
        "completion_0": "번디는 다큐멘터리 방송 감독티비 에게 말했다.",
        "completion_1": "라이언에게 말했다.",
        "completion_2": "It is said that the doctor said to her: \"You are in a condition that makes you harmful to yourself and others, but do not worry, I will take care of you.\" \n\n\"I am not worried about myself, why should I be?\" \n\"Because you are aLily and should as much as possible avoid touching yourself in order to protect yourself.\"",
        "ranking": [
            1,
            0,
            2
        ]
    }, ...
```
- 사람이 labeling 시 문장을 읽고 ranking을 0~2로 순위를 매긴다
- 향후 Step2) RM 모델을 학습할 때는 아래 형식으로 ranking을 2개씩 묶어 chosen과 rejected 데이터셋으로 변환하여 사용함
```
data = {}
data['prompt'] = 'prompt'
data['chosen'] = 'good_sentence'
data['rejected'] = 'bad_sentence'
```

#### 1.4)  step3, PPO(질문에 ``더`` 잘 답하는 모델) 데이터셋 구축
- PPO 데이터: [**data_kochatgpt/kochatgpt_3_PPO.jsonl**](https://github.com/airobotlab/KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl): 12,000개
- AI가 자동으로 글을 생성하기 위한 prompt 데이터셋
- SFT 데이터셋에서 prompt만 가져와서 jsonl 형태로 변형후 저장
```json
[
    {
        "prompt": ""
    },
    {
        "prompt": ""
    }, ...    
]
```

* * *
* * *

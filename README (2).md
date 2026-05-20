# 🎤 Early Speaker Recognition
> **"음성을 끝까지 듣지 않아도 화자를 맞출 수 있는가?"**

음성의 초기 구간(0.5초 ~ 전체)만으로 화자를 식별하고, 모델별·시간별 성능을 비교한 실험 프로젝트입니다.

---

## 📌 프로젝트 목표

실시간 시스템(전화, AI 스피커 등)에서는 음성 전체를 기다릴 수 없습니다.  
**"최소 몇 초의 음성이 있어야 화자를 신뢰할 수 있는 수준으로 인식할 수 있는가"** 를 실험적으로 분석합니다.

---

## 🧠 배경지식

### 화자 인식 (Speaker Recognition)이란?

화자 인식은 음성 신호로부터 **누가 말하고 있는지** 를 식별하는 기술입니다.  
크게 두 가지로 나뉩니다.

| 종류 | 설명 | 예시 |
|---|---|---|
| Speaker Identification | 등록된 N명 중 누구인지 분류 | 이 프로젝트 (49-class) |
| Speaker Verification | 특정 화자가 맞는지 1:1 검증 | 음성 잠금 해제 |

### 왜 "초기 구간(Early Segment)"인가?

실시간 시스템에서는 발화가 끝날 때까지 기다릴 수 없습니다.

```
전화 연결 즉시 → 0.5초 → 1초 → 2초 → ...
                    ↑
               "이 시점에 이미 누군지 알 수 있는가?"
```

음성은 시간이 지날수록 정보가 누적되므로, **얼마나 짧은 시간에 신뢰할 수 있는 판단을 내릴 수 있는가** 가 실용적 핵심 질문입니다.

### 음성 특징 추출 — Log-Mel Spectrogram

원시 음성 파형(waveform)을 그대로 모델에 입력하기보다, **인간의 청각 특성을 반영한 주파수 표현** 으로 변환합니다.

```
WAV (파형)
  → STFT (Short-Time Fourier Transform)   : 시간-주파수 표현
  → Mel Filter Bank 적용                  : 인간 청각 척도(Mel scale)로 변환
  → Log 변환                              : 지각적 크기 압축
  → Log-Mel Spectrogram [64 mel, T frames]
```

- **Mel scale**: 인간은 저주파 변화에 더 민감하므로, 낮은 주파수를 더 세밀하게 표현
- **Log 변환**: 작은 소리와 큰 소리의 크기 차이를 압축해 학습 안정화
- **hop_length=160, sr=16,000** → 1초 = 약 100 프레임

### TCN (Temporal Convolutional Network)

TCN은 RNN 계열 대신 **Dilated Causal Convolution** 으로 시계열을 처리합니다.

```
dilation=1 : [■ ■ ■]               → 인접 3프레임 참조
dilation=2 : [■ · ■ · ■]           → 2칸 간격 참조
dilation=4 : [■ · · · ■ · · · ■]   → 4칸 간격 참조
dilation=8 : 매우 넓은 범위 참조
```

- **장점**: 병렬 처리 가능 (RNN 대비 빠름), 긴 시간 범위를 효율적으로 참조
- **이 프로젝트**: dilation 1→2→4→8, 4개 TemporalBlock 사용

---

## 📂 데이터

| 항목 | 내용 |
|---|---|
| 출처 | [AI Hub — 한국어 음성 대화 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=537) |
| 원본 규모 | WAV + JSON 각 4,835개 |
| 사용 화자 수 | 49명 (발화 수 ≥ 3인 화자만 유지) |
| 샘플링 레이트 | 16,000 Hz |
| 최종 샘플 수 | 21,674개 (chunk별 중복 포함) |

### 데이터 분할

| Split | 비율 | 기준 |
|---|---|---|
| Train | 70% | 화자 단위 분리 (화자 누수 방지) |
| Validation | 15% | |
| Test | 15% | |

> **화자 누수(Speaker Leakage)란?**  
> 같은 화자의 발화가 train과 test에 동시에 들어가면 모델이 "이 목소리를 들어본 적 있다"는 정보를 쓰게 돼 정확도가 과대평가됩니다.  
> 이를 방지하기 위해 **화자 단위**로 분리합니다.

---

## 🔧 전처리 파이프라인

### 전체 흐름

```
원본 WAV + JSON
      ↓
① JSON 파싱 (SpeechStart / SpeechEnd 추출)
      ↓
② 발화 구간 자르기 (무음 제거)
      ↓
③ 시간별 chunk 생성 (0.5s / 1.0s / 2.0s / full)
      ↓
④ Log-Mel Spectrogram 추출 → .npy 저장
      ↓
⑤ 짧은 발화 필터링
      ↓
⑥ 화자 균형 조정
      ↓
⑦ 화자 단위 Train / Val / Test 분리
      ↓
final_manifest.csv (21,674행)
```

### 각 단계 상세

**① JSON 파싱**

AI Hub 데이터의 JSON에는 발화 메타 정보가 포함되어 있습니다.

```json
{
  "NumberOfSpeaker": "2636",
  "FileName": "B0129-2636F0010.wav",
  "SpeechStart": "0.4",
  "SpeechEnd": "1.44"
}
```

화자 ID, 파일 경로, 발화 시작/종료 시점을 파싱해 매핑합니다.

**② 발화 구간 자르기**

WAV 파일 전체가 아닌 `SpeechStart ~ SpeechEnd` 구간만 사용합니다.  
앞뒤 무음이 포함되면 모델이 침묵 패턴을 학습할 수 있어 제거합니다.

```
전체 WAV:  [무음][발화 구간][무음]
사용 구간:       [발화 구간]
```

**③ 시간별 chunk 생성 (핵심)**

동일한 발화를 시간별로 잘라서 여러 버전의 데이터를 생성합니다.  
이렇게 하면 **"같은 발화 내에서 시간만 다르게 비교"** 가 가능해집니다.

```python
clip = audio[speech_start : speech_end]

clip_0_5s = clip[:0.5초]   # label: 화자ID
clip_1_0s = clip[:1.0초]   # label: 화자ID
clip_2_0s = clip[:2.0초]   # label: 화자ID
clip_full = clip            # label: 화자ID
```

| Chunk | 프레임 수 | 샘플 수 |
|---|---|---|
| 0.5s | 50 | ~4,824 |
| 1.0s | 100 | ~4,778 |
| 2.0s | 200 | ~3,249 |
| full | 300 | ~4,833 |

**④ Log-Mel Spectrogram 추출**

```python
mel = librosa.feature.melspectrogram(y, sr=16000, n_mels=64, hop_length=160)
log_mel = librosa.power_to_db(mel)  # shape: [64, T]
np.save(output_path, log_mel)
```

- 64차원 mel filterbank → 시간 축 T는 chunk 길이에 따라 달라짐
- `.npy` 형식으로 저장해 학습 시 빠른 로딩

**⑤ 짧은 발화 필터링**

발화 길이가 chunk window보다 짧으면 제거합니다.  
예: 0.5s chunk를 만들려면 발화가 최소 0.5초 이상이어야 함

**⑥ 화자 균형 조정**

특정 화자가 데이터를 지배하지 않도록 **화자당 최대 15개 파일** 로 제한합니다.

**⑦ 정규화**

학습 시 샘플별로 평균과 표준편차를 이용한 정규화를 적용합니다.

```python
feat = (feat - feat.mean()) / (feat.std() + 1e-6)
```

---

## 🤖 모델

### 1. LightGBM (Baseline)

- **입력**: log-mel → `mean(axis=1) + std(axis=1)` → **128차원 벡터**
- 시간 흐름을 완전히 무시하고 통계값만 사용
- "시간 정보를 쓰지 않으면 어느 수준인가"의 대조군

```
log-mel [64, T] → mean 64차원 + std 64차원 → 128차원 벡터 → LightGBM
```

### 2. 1D-CNN

- **입력**: log-mel `[64, T]` 그대로 사용
- Conv1d × 3 + BatchNorm + ReLU + Dropout + MaxPool
- AdaptiveAvgPool → Linear(256) → Linear(num_classes)
- 지역적 시간 패턴 추출에 특화

```
[64, T] → Conv(128) → MaxPool → Conv(256) → MaxPool → Conv(256) → AvgPool → FC
```

### 3. TCN (Temporal Convolutional Network)

- **입력**: log-mel `[64, T]` 그대로 사용
- TemporalBlock × 4 (dilation: 1 → 2 → 4 → 8)
- Residual connection + Causal padding
- 장기 시계열 의존성 모델링에 특화

```
[64, T] → TB(dil=1) → TB(dil=2) → TB(dil=4) → TB(dil=8) → AvgPool → FC
```

---

## ⚙️ 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | val_acc 기준 patience=7 |
| Max Epochs | 20 |
| Batch Size | 32 |
| Loss | CrossEntropyLoss |
| Device | GPU (T4) |
| Seed | 42 |

---

## 📊 실험 결과

### Test Accuracy

| 모델 | 0.5s | 1.0s | 2.0s | full |
|---|---|---|---|---|
| LightGBM | 0.644 | 0.764 | 0.787 | 0.874 |
| 1D-CNN | 0.800 | **0.878** | 0.891 | 0.919 |
| **TCN** | 0.793 | 0.855 | **0.909** | **0.942** |

### Test Macro F1

| 모델 | 0.5s | 1.0s | 2.0s | full |
|---|---|---|---|---|
| LightGBM | 0.589 | 0.726 | 0.682 | 0.821 |
| 1D-CNN | 0.768 | **0.857** | 0.806 | 0.890 |
| **TCN** | 0.762 | 0.819 | **0.832** | **0.907** |

---

## 📈 핵심 결과 시각화

![꺾은선 그래프](results/key_result.png)
![막대 그래프](results/model_chunk_comparison.png)

---

## 🔍 결과 분석

### 1. 시간이 길수록 성능이 향상된다
모든 모델에서 chunk 길이가 길어질수록 정확도가 단조 증가합니다.  
0.5s → full 구간에서 TCN 기준 **79.3% → 94.2%** 로 약 15%p 향상됩니다.

### 2. 1초가 핵심 변곡점이다
0.5s → 1.0s 구간에서 딥러닝 모델의 정확도가 가장 급격히 상승합니다.  
1초 이상부터 화자 식별에 필요한 기본 음성 정보가 확보되는 것으로 해석됩니다.

### 3. TCN 기준 2초면 90% 달성 — 실용적 최소 구간
TCN은 2.0s에서 **90.9%** 를 달성하며 90% 기준선을 돌파합니다.  
실시간 시스템에서 **2초** 만 수집해도 충분히 신뢰할 수 있는 화자 인식이 가능합니다.

### 4. 짧은 음성에서는 단순 모델이 유리할 수 있다
1.0s 구간에서 1D-CNN(87.8%)이 TCN(85.5%)을 앞섭니다.  
TCN의 넓은 dilation 구조가 짧은 음성에서는 오히려 과도한 컨텍스트를 참조하며 불리하게 작용한 것으로 해석됩니다.  
음성이 길어질수록(2.0s~) TCN의 장기 의존성 모델링 강점이 드러나며 역전됩니다.

### 5. 시간 정보 활용 여부가 성능을 가른다
LightGBM은 시간 정보를 mean/std로 압축하기 때문에 전 구간에서 딥러닝 대비 열위입니다.  
특히 짧은 구간일수록 격차가 크며, 이는 **화자 정보가 시간 흐름 자체에 담겨 있음** 을 시사합니다.

---

## 💡 활용 가능 분야

- **음성 인증 시스템** — 은행 ARS, 스마트폰 잠금 해제
- **AI 스피커 개인화** — 가족 구성원 자동 구분
- **회의 자동 기록** — 화자 분리 기반 회의록 생성
- **콜센터 분석** — 상담원 / 고객 자동 구분
- **음성 포렌식** — 녹음 파일 화자 식별

---

## 📁 디렉터리 구조

```
📦 프로젝트 루트
├── preprocessed_early_speaker/
│   ├── features_logmel/        # log-mel .npy 특징 파일
│   └── meta/
│       ├── final_manifest.csv  # 전체 샘플 메타데이터
│       └── speaker_label_map.csv
├── checkpoints/
│   ├── LightGBM_{chunk}/
│   │   └── cm_LightGBM_{chunk}.png
│   ├── 1D-CNN_{chunk}/
│   │   ├── best_model.pt
│   │   ├── cm_1D-CNN_{chunk}.png
│   │   └── curve_1D-CNN_{chunk}.png
│   └── TCN_{chunk}/
│       ├── best_model.pt
│       ├── cm_TCN_{chunk}.png
│       └── curve_TCN_{chunk}.png
├── results/
│   ├── progress.csv              # 전체 실험 결과
│   ├── key_result.png            # 핵심 꺾은선 그래프
│   └── model_chunk_comparison.png
└── train_colab_final.ipynb       # 전체 실험 노트북
```

---

## 🚀 실행 방법

### 환경
- Python 3.10+
- PyTorch, LightGBM, librosa, scikit-learn, seaborn

### Google Colab에서 실행

```python
# 1. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. 패키지 설치
!pip install lightgbm seaborn -q

# 3. train_colab_final.ipynb 열어서 순서대로 실행
```

### 이어서 실행 (중단된 경우)

실험 도중 세션이 끊겨도 `results/progress.csv` 에 완료된 실험이 저장되어 있어  
노트북을 처음부터 다시 실행하면 완료된 조합은 자동으로 건너뜁니다.

---

## 📝 결론

> **TCN 기준 2초의 음성만으로 90% 이상의 화자 인식 정확도를 달성할 수 있다.**  
> 짧은 음성에서는 단순한 모델이, 긴 음성에서는 시계열 특화 모델이 더 효과적이며,  
> 시간 정보를 직접 활용하는 딥러닝 접근이 tabular 모델 대비 일관되게 우수한 성능을 보인다.

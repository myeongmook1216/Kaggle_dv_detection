# HBNU Fake Audio Detection Competition

**기간:** 2024년 5월 1일 - 2024년 5월 17일  
**대회 링크:** [Kaggle HBNU Fake Audio Detection Competition](https://www.kaggle.com/competitions/hbnu-fake-audio-detection-competition/submissions)

해당 대회는 Deep Voice의 진위여부를 판단하는 대회로, 총 8000개의 train data를 이용하였습니다. (True: 4000개, False: 4000개)

## 1. Pre-processing

### Data Augmentation
- **기법:** time_stretch와 pitch_shift
- **설명:** wav 파일의 audio 값과 sampling rate 값을 추출하여 위의 기법을 활용해 데이터를 증강했습니다.

### Feature Extraction
- **기법:** Mel-spectrogram
- **크기:** 100 x 100
- **설명:** 증강된 데이터로부터 Mel-spectrogram을 (100, 100) 크기로 추출했습니다. 후에 모델에 적합한 크기로 넣기 위해 (32, 1, 384, 384)로 차원을 맞추어 주었습니다.
  - **384 x 384:** 모델과의 적합성을 고려하여 결정했습니다.
  - **32:** 실험을 통해 batch_size는 32가 적합하다고 판단되었습니다.
  - **1:** mfcc와 mel을 동시에 추출하여 2개의 차원으로 넣어보았으나, mel_spectrogram만 추출하여 단일 차원으로 넣는 것이 더 높은 정확도를 보였습니다.

## 2. 모델 정보
- Resnet18, 34, 50, EfficientNetV2_S 등을 사용해 보았지만, 원하는 결과를 얻기 힘들어서 직접 모델을 구현하였습니다.
- Residual Block을 이용하여 모델을 구성한 뒤에, 실험을 통해 적절한 넓이와 깊이를 가지는 모델을 찾아내었습니다.
- Complexity를 추가하기 위해 Residual Block에 SE Block을 결합한 뒤에 적절히 섞어보았지만 유의미한 성능지표를 보이지는 않았습니다.

## 3. 피드백
- **모델:** 데이터의 양에 적절한 깊이와 넓이, 복잡성을 가지는 모델을 찾아내는 것이 오래 걸렸습니다.
- **특징 추출:** 다양한 특성 지표 중 mel_spectrogram, mfcc, chroma가 deep voice의 진위여부 판단에 유의미한 결과를 가져오는 특성임을 찾아내었습니다.
  - 동일 모델에 3가지 특징을 각각 적용시켜 보았을 때 mel > mfcc > chroma 순으로 정확도가 측정되었습니다.
  - mel과 mfcc를 각각 학습한 모델에 대하여 소프트 보팅을 진행하였으나 성능 향상을 가져오지 못했습니다.
  - mel과 mfcc를 각 차원으로 보고 (32, 2, 384, 384)로 만들어서 모델에 넣어보았으나 성능 향상을 가져오지 못했습니다. (여기서 2는 각각 동일한 wav에서 뽑아낸 mel, mfcc)

## 참고 논문
- [REAL-TIME DETECTION OF AI-GENERATED SPEECH FOR DEEPFAKE VOICE CONVERSION](https://arxiv.org/pdf/2308.12734)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)

## 소감
논문을 읽고 이론을 안다고 해서 딥러닝을 아는 것이 아니라는 것을 이번 대회를 통해 깨달았습니다. 논문에서 필요한 내용만 발췌하여 코드를 저의 코드에 직접 적용시켜보고 그것을 통해 지표를 뽑아내는 과정이 저의 앞으로의 공부 방식에 있어서 적합하다는 것을 깨달았습니다. 짧은 대회 기간이었지만, 해당 대회를 진행하며 딥러닝 모델의 전반적인 이해도가 많이 향상되었음을 느꼈고, 앞으로의 방향성을 잡을 수 있게 하는 계기가 되었습니다.

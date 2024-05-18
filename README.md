24.05.01 - 24.05.17
해당 기간동안 캐글에서 대회를 진행하였습니다. https://www.kaggle.com/competitions/hbnu-fake-audio-detection-competition/submissions

해당 대회는 deep voice의 진위여부를 판단하는 대회로써, 총 8000개의 train data를 이용하였습니다.( 4000 - True, 4000 - False )

1. pre-processing
- data augmentation
- wav 파일의 audio 값과 sampling rate 값을 추출하여 time_stretch와 pitch_sift 기법을 활용하여 증강을 하였습니다.
- feature extraction
- 증강된 데이터에 대하여 mel_spectrogram을 (100, 100)크기로 추출하였습니다.
- 후에, 모델에 적합한 크기로 넣기 위해 (32, 1, 384, 384)로 차원을 맞추어 주었습니다.
- (384, 384) - 모델과의 적합성 고려
- 32 - 실험을 통해 batch_size는 

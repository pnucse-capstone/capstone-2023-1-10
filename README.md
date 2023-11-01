### 1. 프로젝트 소개

신약 개발을 위한 binding affinity prediction

본 과제에선 연구는 두 개의 CNN(Convolution Neural Network)을 지닌 DeepDTA 모델을 구현하고 단백질 서열과 리간드 SMILES를 학습시켜 약물-표적 결합 친화도를 예측하는 것을 목표로 한다.

### 2. 팀 소개

- 임연후 yeonhu82g@gmail.com

  - 학습 데이터셋 탐색
  - Pytorch를 이용한 CNN 블록 구현
  - FC layer모델 구현

- 박한얼 hanul.park@gmail.com

  - 개발 환경 구축
  - 데이터 전처리
  - Hyperparameter 최적화

- 김선아 llksall@pusan.ac.kr
  - 학습 데이터셋 탐색
  - input dataset 훈련모델(Trainer) 구현
  - 모델 성능 측정

### 3. 구성도

_DeepDTA 구조_
![DEEPDTA](https://github.com/pnucse-capstone/Capstone-Template-2023/assets/71930280/a31731d5-aefa-4610-ad17-d290303e4ca3)

#### 사용 데이터 셋

![데이터셋](https://github.com/pnucse-capstone/Capstone-Template-2023/assets/71930280/b645ef67-3e41-48cf-9f0d-d4a8191ef855)

#### 하이퍼파라이터 그리드

![run_experiment.py](https://github.com/pnucse-capstone/Capstone-Template-2023/assets/71930280/e9307622-e377-46be-b9e0-7e7af6d2d963)

#### 모델과 Trainer

![image](https://github.com/pnucse-capstone/Capstone-Template-2023/assets/71930280/f2e31e44-dd2a-4a76-8070-6ac41617bf26)

#### 출력 로그

![로그](https://github.com/pnucse-capstone/Capstone-Template-2023/assets/71930280/5c261976-8160-4e4d-a2d6-9f7f1b35687b)

### 4. 소개 및 시연 영상

-소개 영상 : https://youtube.com/watch?v=raNwONzdBWE&si=bydRqVApjQgaakod

### 5. 사용법

> **프로그램 최소 요구 사항**  
> python 3.7.9  
> pandas 1.3.5  
> numpy 1.21.6  
> tqdm 4.66.1  
> scikit-learn 1.0.2

**실행**  
run_experiment.py 파일 실행

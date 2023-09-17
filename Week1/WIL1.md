1. AI, ML, DL의 차이와 관계
AI(Artificial Inteligence) : 인간의 능력을 인공적으로 구현
ML(Machine Learning) : 데이터로부터 규칙을 학습하는 AI의 하위 분야
DL(Deep Learning) : Neural network를 기반으로 한 ML의 하위분야

2. DL Component
1) Data : 풀고자하는 task에 따라서 다르게 제공되어야 한다.(dependent)
- Classification : Classification dataset
- Semantic Segmentation : Semantic Segmentation dataset
- Object Detection : bbox (x, y), w, h
- Pose Estimation : 3d, 2d skeleton dataset
2) Model : input에서 feature를 뽑고 우리가 원하는 output으로 만드는 프로그램
- AlexNet, GoogLeNet, ResNet, DenseNet, LSTM, AutoEncoder, GAN
3) Loss function : 학습 중 알고리즘이 얼만큼 잘못 예측하나에 대한 지표
- 알고리즘이 예측한 값과 실제 정답의 차이를 비교해서 학습한다
4) Optimization and Regulatiztion
- Optimization
Gradient Descent Method(경사하강법) : loss function을 빠르고 정확하게 줄이기 위한 최적화 기법
보통 Adam이나 AdamW 사용함
- Regulatization
학습을 의도적으로 방해하여 일반화 성능 up

3. Neural Network
Neural Networks are Function approximators that stack affine transformations followed by nonlinear transformations
Matrix multiplication
nonlinear function

4. Nonlinear Function
활성함수(Activation function)으로 비선형 함수를 사용하는 이유 : 신경망의 표현성을 높일 수 있음
활성함수가 선형함수라면 여러 겹을 쌓아도 하나의 활성함수로 대체 가능하다.

5. Multi-Layer Perceptron
- Loss function의 모순

6. Generalization
- 일반화 성능을 높이는데에 목적
- Generalization Gap = | Test error - Training error |
- Under-fitting(과소적합, 지나치게 단순하여 일반성 X) < Optimal Balance(밸런스 굿) < Over-fitting(과대적합, 지나치게 복잡하여 일반성 X)
- Cross Validation(교차검증)
test data를 학습시에 사용하는 것은 cheating으로 간주됨. 그러나 학습이 정상적으로 진행되는지 확인하기 위해서 valid data를 사용.
- Ensemble(앙상블 기법) : 여러개의 분류모델을 조합하여 성능을 향상시킨다.
Bagging : subset을 나누어 학습 후 각각의 voting이나 averaging을 구한다.(병력적으로 학습 = parallel)
Boosting : 학습이 제대로 되지 않은 데이터들을 모아서 새로운 간단한 모델로 재학습.(연속적으로 학습=squential)
- Regulatization(학습을 방해하기)
Early stopping
Parameter norm penalty
Data augmentation
Noise robustness : 노이즈와 같은 간섭이 있어도 흔들리지 않도록
Dropout : train에서만 적용되는 방법으로 test에서는 모든 노드가 추론에 참여

7. Convolutional Neural Networks(CNN) : 합성곱 신경망
Convolution 계산과정 - > 영상과 그림으로 익히기 / 파이썬에서 짜보기(W1과제)
RGB 계산

8. 1x1 Convolution : depth차원을 변경할 수 있다. neural netowrk를 깊게 쌓을 수 있음.

9. Modern CNN
1) AlexNet(2012) : 2 Networks(GPU issue), 5 convolution layers, 3 dense layers (W1과제에서 해보자)
2) VGGNet(2014) : 3x3 convolution filter : 같은 receptive filed에서 계산량을 줄인다.
3) GoogLeNet(2014) : 1x1 convolution
4) ResNet(2015) : 사람의 능력을 뛰어넘은 첫번째 모델 = skip connection = shortcut = residual gradient : 미분해도 기울기 1 남으므로 소실 문제 해결

#
딥러닝의 기초 이론을 배웠다. 몇번은 더 읽어봐야 할 것 같다. 일단 과제 풀고, 나름 끄적여봐야겠다.
<<<<<<< HEAD
# Computer-Vision
=======
# 이미지 전처리 프로젝트

## 개요
이 프로젝트는 Hugging Face의 COCO 데이터셋에서 이미지를 로드하여 컴퓨터 비전 작업을 위한 전처리를 수행합니다.

## 필요한 라이브러리
```bash
pip install opencv-python numpy datasets
```

## 전처리 과정

### 1. 데이터셋 로드
```python
ds = load_dataset("detection-datasets/coco", split="train[:6]")
```
- Hugging Face에서 COCO 데이터셋의 학습 데이터 처음 5개 샘플 로드

### 2. 이미지 형식 변환
```python
image_pil = ds['image'][i]  # PIL Image 형식
image_rgb = np.array(image_pil)  # numpy 배열로 변환 (RGB)
image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # BGR로 변환 (OpenCV 기본 형식)
```
- PIL Image → Numpy 배열 → BGR 형식 변환

### 3. 리사이징
```python
image = cv2.resize(image, (224, 224))
```
- 이미지를 224×224 크기로 통일
- 모델 입력 크기 표준화

### 4. Grayscale 변환
```python
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
- 컬러 이미지를 흑백 이미지로 변환
- 채널 수 감소로 계산량 감소

### 5. 정규화 (Normalization)
```python
image_normalized = image_gray.astype(np.float32) / 255.0
```
- 픽셀 값을 0~1 범위로 정규화
- 머신러닝 모델의 안정적인 학습 지원

### 6. 가우시안 블러 (노이즈 제거)
```python
image_blurred = cv2.GaussianBlur(image_normalized, (5, 5), 0)
```
- 5×5 커널로 가우시안 블러 필터 적용
- 이미지 노이즈 감소

### 7. 데이터 증강 - 좌우 반전
```python
image_flipped = cv2.flip(image_blurred, 1)
```
- 이미지를 수평축 기준으로 반전 (좌우 반전)
- 훈련 데이터 다양성 증가

### 8. 데이터 증강 - 90도 회전
```python
rows, cols = image_flipped.shape
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
image_rotated = cv2.warpAffine(image_flipped, rotation_matrix, (cols, rows))
```
- 회전 행렬을 이용하여 이미지를 90도 회전
- 모델의 회전 불변성 향상

## 처리 결과 확인
```python
cv2.imshow('Original', image)  # 원본 이미지 (리사이징만 적용)
cv2.imshow('Preprocessed', image_rotated)  # 최종 전처리된 이미지
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 창 닫기
```

## 전처리 파이프라인 요약

```
원본 이미지 (PIL)
    ↓
RGB → BGR 변환
    ↓
224×224 리사이징
    ↓
Grayscale 변환
    ↓
정규화 (0~1)
    ↓
가우시안 블러 (노이즈 제거)
    ↓
좌우 반전 (데이터 증강)
    ↓
90도 회전 (데이터 증강)
    ↓
최종 전처리 완료
```

## 주요 기능

| 기능 | 목적 | 효과 |
|------|------|------|
| 리사이징 | 입력 크기 표준화 | 모델 호환성 확보 |
| Grayscale | 채널 감소 | 계산량 감소 |
| 정규화 | 값 범위 조정 | 모델 학습 안정화 |
| 블러 | 노이즈 제거 | 이미지 품질 개선 |
| 반전/회전 | 데이터 증강 | 모델 성능 향상 |

## 실행 방법

```bash
python image_preprocessing.py
```

프로그램 실행 후 각 이미지마다 원본과 전처리된 이미지가 창에 표시됩니다.
원본 이미지 창에서 아무 키나 누르면 다음 이미지로 진행합니다.
>>>>>>> 4641b21 (Added image preprocessing script)

import cv2
import numpy as np
from datasets import load_dataset


#### 이미지 전처리 ####

# Hugging Face 데이터셋에서 이미지 로드
ds = load_dataset("detection-datasets/coco", split="train[:1]")

# 이미지 가져오기 (PIL Image 형식)
image_pil = ds['image'][0]

# PIL Image를 numpy 배열로 변환 (RGB 형식)
image_rgb = np.array(image_pil)

# RGB를 BGR로 변환 (OpenCV는 BGR 사용)
image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# [기본 과제] 이미지 전처리
image = cv2.resize(image, (224, 224)) # 크기 조정 (224*224)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 색상 변환 (Grayscale 적용)
image_normalized = image_gray.astype(np.float32) / 255.0 # 색상 변환 (Normalize 적용)
image_blurred = cv2.GaussianBlur(image_normalized, (5, 5), 0) # 노이즈 제거 (Blur 필터 적용)
image_flipped = cv2.flip(image_blurred, 1) # 데이터 증강 (좌우 반전)

# 데이터 증강 (90도 회전)
rows, cols = image_flipped.shape 
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
image_rotated = cv2.warpAffine(image_flipped, rotation_matrix, (cols, rows))


#### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성 ####

# Depth Map 
image_rotated_uint8 = (image_rotated * 255).astype(np.uint8)
depth_map = cv2.applyColorMap(image_rotated_uint8, cv2.COLORMAP_JET)

# 3D 포인트 클라우드 변환
h, w = depth_map.shape[:2]
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = image_rotated.astype(np.float32) # Depth 값을 Z 축으로 사용

# 3D 좌표 생성
points_3d = np.dstack((X, Y, Z))

# 결과 출력
cv2.imshow('Preprocessed', image_rotated)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
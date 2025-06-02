#!/usr/bin/env python3
import os
import cv2
import numpy as np
import math

def get_rgb_thresh_img(img, channel='R', thresh=(0, 255)):
    """RGB 색상 공간에서 특정 채널에 대한 이진화 이미지 생성"""
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel == 'R':
        bin_img = img1[:, :, 0]
    elif channel == 'G':
        bin_img = img1[:, :, 1]
    elif channel == 'B':
        bin_img = img1[:, :, 2]

    binary_img = np.zeros_like(bin_img).astype(np.uint8)
    binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1

    return binary_img

def get_lab_bthresh_img(img, thresh=(0, 255)):
    """LAB 색상 공간에서 B 채널에 대한 이진화 이미지 생성"""
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = lab_img[:, :, 2]

    bin_op = np.zeros_like(B).astype(np.uint8)
    bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1

    return bin_op

def get_hls_sthresh_img(img, thresh=(0, 255)):
    """HLS 색상 공간에서 채도(S) 채널에 대한 이진화 이미지 생성"""
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S).astype(np.uint8)
    binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1

    return binary_output

def get_bin_img(img, kernel_size=5, sobel_thresh=(30, 200), 
                r_thresh=(170, 255), s_thresh=(120, 255), 
                b_thresh=(200, 255), g_thresh=(220, 255)):
    """여러 색상 공간과 소벨 필터를 조합하여 이진화 이미지 생성"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 소벨 필터 적용
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 소벨 이진화
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    # 각 색상 채널별 이진화
    r_binary = get_rgb_thresh_img(img, thresh=r_thresh)
    g_binary = get_rgb_thresh_img(img, channel='G', thresh=g_thresh)
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # 모든 채널 조합
    combined_binary = np.zeros_like(sbinary)
    combined_binary255 = np.zeros_like(sbinary)
    combined_binary[(r_binary == 1) | (sbinary == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 1
    combined_binary255[(r_binary == 1) | (sbinary == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 255

    return combined_binary, combined_binary255

def transform_image(img, offset=250):
    """원근 변환을 통해 이미지를 버드아이 뷰로 변환"""
    img_size = (img.shape[1], img.shape[0])
    
    # 기본 변환 좌표 설정 (시뮬레이션용)
    leftupper = (270, 250)
    rightupper = (350, 250)
    leftlower = (-300, img.shape[0])
    rightlower = (940, img.shape[0])

    warped_leftupper = (offset, 0)
    warped_rightupper = (offset, img.shape[0])
    warped_leftlower = (img.shape[1] - offset, 0)
    warped_rightlower = (img.shape[1] - offset, img.shape[0])

    src = np.float32([leftupper, leftlower, rightupper, rightlower])
    dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

    # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 이미지 워핑
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    
    return warped, M

def find_lanes(binary_warped, nwindows=15, margin=30, minpix=50):
    """슬라이딩 윈도우 방식으로 차선 픽셀 검출"""
    # 이미지 아래쪽 절반의 히스토그램 계산
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    
    # 시각화를 위한 출력 이미지 생성
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # 히스토그램의 왼쪽과 오른쪽 절반에서 피크 찾기
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 윈도우 높이 설정
    window_height = int(binary_warped.shape[0] // nwindows)
    
    # 이미지에서 0이 아닌 모든 픽셀의 위치 식별
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # 각 윈도우의 현재 위치 초기화
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # 왼쪽 및 오른쪽 차선 픽셀 인덱스를 저장할 빈 리스트 생성
    left_lane_inds = []
    right_lane_inds = []
    
    # 윈도우를 하나씩 처리
    for window in range(nwindows):
        # 윈도우의 y 경계 식별
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        # 윈도우의 x 경계 식별
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # 시각화 이미지에 윈도우 그리기
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)
        
        # 윈도우 내의 0이 아닌 픽셀 식별
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # 인덱스를 리스트에 추가
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # 최소 픽셀 수보다 많은 픽셀을 찾았으면 다음 윈도우 중심 위치 업데이트
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    # 인덱스 배열 연결
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # 구현이 완전하지 않은 경우 오류 방지
        pass
    
    # 왼쪽 및 오른쪽 라인 픽셀 위치 추출
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # 2차 다항식 피팅
    left_fit = np.array([0, 0, 0]) if lefty.size == 0 and leftx.size == 0 else np.polyfit(lefty, leftx, 2)
    right_fit = np.array([0, 0, 0]) if rightx.size == 0 and righty.size == 0 else np.polyfit(righty, rightx, 2)
    
    # 플롯팅을 위한 x, y 값 생성
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
    
    # 왼쪽 및 오른쪽 차선 영역에 색상 표시
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 255, 0]
    
    # 피팅된 다항식 시각화
    for y, _ in enumerate(left_fitx):
        left_val = min(max(int(left_fitx[y]), 0), out_img.shape[1]-1)
        right_val = min(max(int(right_fitx[y]), 0), out_img.shape[1]-1)
        out_img[y, left_val] = [255, 234, 0]
        out_img[y, right_val] = [255, 234, 0]
    
    return out_img, left_fit, right_fit, left_fitx, right_fitx

def process_frame(frame):
    """프레임을 처리하여 차선 검출"""
    # 이미지 이진화
    binary, binary255 = get_bin_img(frame)
    
    # 원근 변환
    warped, M = transform_image(binary)
    
    # 차선 검출
    lane_img, left_fit, right_fit, left_fitx, right_fitx = find_lanes(warped)
    
    # 차선 검출 이미지만 반환
    return lane_img
    

    # # 결과 이미지 합치기 (원본, 이진화, 차선 검출)
    # h, w = frame.shape[:2]
    
    # # 이미지 크기 조정 (원본 프레임)
    # frame_resized = cv2.resize(frame, (w//2, h//2))
    
    # # 이진화 이미지 크기 조정 및 3채널로 변환
    # binary255_resized = cv2.resize(binary255, (w//2, h//2))
    # binary255_colored = np.dstack((binary255_resized, binary255_resized, binary255_resized))
    
    # # 차선 검출 이미지 크기 조정
    # lane_img_resized = cv2.resize(lane_img, (w//2, h//2))
    
    # # 이미지 합치기
    # top_row = np.hstack((frame_resized, binary255_colored))
    # bottom_row = np.hstack((lane_img_resized, np.zeros_like(frame_resized)))  # 빈 공간으로 채움
    
    # result = np.vstack((top_row, bottom_row))
    
    # # 각 창에 레이블 추가
    # cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(result, "Binary", (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(result, "Lane Detection", (10, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # return result

def main():
    input_path = 'input/1.mp4'
    
    video_name = os.path.basename(input_path).split('.')[0]
    
    output_base = 'output'
    output_dir = os.path.join(output_base, video_name)
    
    folder_counter = 0
    while os.path.exists(output_dir):
        folder_counter += 1
        output_dir = os.path.join(output_base, f"{video_name}_{folder_counter}")

    if not os.path.exists(output_base):
        os.makedirs(output_base)
        
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Can't open the video.")
        return
    
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video end or error occurred.")
            break
        
        result = process_frame(frame)
        
        cv2.imshow('Lane Detection', result)
        
        cv2.imwrite(f'{output_dir}/frame_{index:04d}.jpg', result)
        index += 1
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"Total {index} frames have been saved to {output_dir}")
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
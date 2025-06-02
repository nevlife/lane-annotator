#!/usr/bin/env python3
import cv2
import numpy as np
import math

class LaneDetection:
    def __init__(self):
        self.src = None
        self.dst = None
    
    def set_perspective_transform(self, src, dst):
        """원근 변환을 위한 소스와 목적지 포인트 설정"""
        self.src = src
        self.dst = dst
    
    def get_rgb_thresh_img(self, img, channel='R', thresh=(0, 255)):
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

    def get_lab_bthresh_img(self, img, thresh=(0, 255)):
        """LAB 색상 공간에서 B 채널에 대한 이진화 이미지 생성"""
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        B = lab_img[:, :, 2]

        bin_op = np.zeros_like(B).astype(np.uint8)
        bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1

        return bin_op

    def get_lab_athresh_img(self, img, thresh=(0, 255)):
        """LAB 색상 공간에서 A 채널에 대한 이진화 이미지 생성"""
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        A = lab_img[:, :, 1]

        bin_op = np.zeros_like(A).astype(np.uint8)
        bin_op[(A >= thresh[0]) & (A < thresh[1])] = 1

        return bin_op

    def get_hls_sthresh_img(self, img, thresh=(0, 255)):
        """HLS 색상 공간에서 채도(S) 채널에 대한 이진화 이미지 생성"""
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        S = hls_img[:, :, 2]

        binary_output = np.zeros_like(S).astype(np.uint8)
        binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1

        return binary_output

    def get_bin_img(self, img, kernel_size=10, sobel_dirn='X', sobel_thresh=(0, 255), 
                    r_thresh=(0, 255), s_thresh=(0, 255), b_thresh=(0, 255), g_thresh=(0, 255)):
        """여러 색상 공간과 소벨 필터를 조합하여 이진화 이미지 생성"""
        # HLS 색상 공간 변환
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 소벨 필터 적용
        if sobel_dirn == 'X':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 소벨 이진화
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

        combined = np.zeros_like(sbinary)
        combined[(sbinary == 1)] = 1

        # 각 색상 채널별 이진화
        r_binary = self.get_rgb_thresh_img(img, thresh=r_thresh)
        g_binary = self.get_rgb_thresh_img(img, thresh=g_thresh, channel='G')
        b_binary = self.get_lab_bthresh_img(img, thresh=b_thresh)
        s_binary = self.get_hls_sthresh_img(img, thresh=s_thresh)

        # 모든 채널 조합
        combined_binary = np.zeros_like(combined)
        combined_binary255 = np.zeros_like(combined)
        combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 1
        combined_binary255[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 255

        return combined_binary, combined_binary255

    def transform_image(self, img, offset=250, src=None, dst=None):
        """원근 변환을 통해 이미지를 버드아이 뷰로 변환"""
        img_size = (img.shape[1], img.shape[0])
        out_img_orig = np.copy(img)

        # 기본 변환 좌표 설정
        leftupper = (265, 200)
        rightupper = (395, 200)
        leftlower = (-135, img.shape[0])
        rightlower = (795, img.shape[0])

        warped_leftupper = (offset, 0)
        warped_rightupper = (offset, img.shape[0])
        warped_leftlower = (img.shape[1] - offset, 0)
        warped_rightlower = (img.shape[1] - offset, img.shape[0])

        # 소스 및 대상 포인트 설정
        if src is not None:
            src = src
        else:
            src = np.float32([leftupper, leftlower, rightupper, rightlower])

        if dst is not None:
            dst = dst
        else:
            dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

        # 변환 행렬 계산
        M = cv2.getPerspectiveTransform(src, dst)
        minv = cv2.getPerspectiveTransform(dst, src)

        # 이미지 워핑
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
        out_warped_img = np.copy(warped)

        return warped, M, minv, out_img_orig, out_warped_img

    def find_lines(self, warped_img, nwindows=9, margin=80, minpix=40):
        """슬라이딩 윈도우 방식으로 차선 픽셀 검출"""
        # 이미지 아래쪽 절반의 히스토그램 계산
        histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)

        # 시각화를 위한 출력 이미지 생성
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # 히스토그램의 왼쪽과 오른쪽 절반에서 피크 찾기
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # 윈도우 높이 설정
        window_height = np.int(warped_img.shape[0] // nwindows)

        # 이미지에서 0이 아닌 모든 픽셀의 위치 식별
        nonzero = warped_img.nonzero()
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
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height

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
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

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

        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

    def fit_polynomial(self, binary_warped, nwindows=5, margin=50, minpix=50):
        """차선 픽셀을 찾고 다항식으로 피팅"""
        # 차선 픽셀 찾기
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = self.find_lines(
            binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

        # 2차 다항식 피팅
        left_fit = np.array([0, 0, 0]) if lefty.size == 0 and leftx.size == 0 else np.polyfit(lefty, leftx, 2)
        right_fit = np.array([0, 0, 0]) if rightx.size == 0 and righty.size == 0 else np.polyfit(righty, rightx, 2)

        # 플롯팅을 위한 x, y 값 생성
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # 피팅에 실패한 경우
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        # 왼쪽 및 오른쪽 차선 영역에 색상 표시
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 255, 0]

        # 피팅된 다항식 시각화
        for y, data in enumerate(left_fitx):
            left_val = out_img.shape[1]-1 if int(left_fitx[y]) >= out_img.shape[1] else int(left_fitx[y])
            left_val = 0 if left_val < 0 else left_val
            right_val = out_img.shape[1]-1 if int(right_fitx[y]) >= out_img.shape[1] else int(right_fitx[y])
            right_val = 0 if right_val < 0 else right_val
            out_img[y, left_val] = [255, 234, 0]
            out_img[y, right_val] = [255, 234, 0]

        return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img

    def search_around_poly(self, binary_warped, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel, margin=80):
        """이전 프레임의 다항식 주변에서 차선 픽셀 검색"""
        # 0이 아닌 픽셀 찾기
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 이전 다항식 주변의 픽셀 검색
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                    left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                        left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                        right_fit[1] * nonzeroy + right_fit[2] + margin)))
        
        # 왼쪽 및 오른쪽 라인 픽셀 위치 추출
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # 2차 다항식 피팅
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # 실제 세계 좌표에 대한 2차 다항식 피팅
        left_lane_indices = np.polyfit(lefty * ymtr_per_pixel, leftx * xmtr_per_pixel, 2)
        right_lane_indices = np.polyfit(righty * ymtr_per_pixel, rightx * xmtr_per_pixel, 2)
        
        return left_fit, right_fit, left_lane_indices, right_lane_indices

    def process_image(self, image):
        """이미지 처리 파이프라인"""
        # 이미지 전처리 매개변수 설정
        kernel_size = 5
        mag_thresh = (30, 200)
        r_thresh = (170, 255)  # 빨강
        s_thresh = (120, 255)  # 채도
        b_thresh = (200, 255)  # 파랑
        g_thresh = (220, 255)  # 초록

        # 이진화 이미지 생성
        combined_binary, combined_binary255 = self.get_bin_img(
            image, kernel_size=kernel_size, sobel_thresh=mag_thresh,
            r_thresh=r_thresh, s_thresh=s_thresh, b_thresh=b_thresh,
            g_thresh=g_thresh
        )

        # 원근 변환 적용
        if self.src is not None and self.dst is not None:
            warped, warp_matrix, unwarp_matrix, _, _ = self.transform_image(
                combined_binary, src=self.src, dst=self.dst
            )
        else:
            warped, warp_matrix, unwarp_matrix, _, _ = self.transform_image(combined_binary)

        # 픽셀 당 미터 단위 설정
        xmtr_per_pixel = 6.7 / 400
        ymtr_per_pixel = 30 / 480

        # 차선 검출 및 다항식 피팅
        left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, lane_img = self.fit_polynomial(
            warped, nwindows=15, margin=30, minpix=50
        )

        return {
            'binary_image': combined_binary255,
            'warped_image': warped,
            'left_fit': left_fit,
            'right_fit': right_fit,
            'left_fitx': left_fitx,
            'right_fitx': right_fitx,
            'lane_image': lane_img,
            'unwarp_matrix': unwarp_matrix
        }

# 사용 예시
if __name__ == "__main__":
    # OpenCV를 사용하여 이미지 로드
    image = cv2.imread('1.mp4')
    
    if image is not None:
        # 차선 감지 객체 생성
        lane_detector = LaneDetection()
        
        # 이미지 처리
        result = lane_detector.process_image(image)
        
        # 결과 시각화
        cv2.imshow('Original Image', image)
        cv2.imshow('Binary Image', result['binary_image'])
        cv2.imshow('Lane Detection', result['lane_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("이미지를 로드할 수 없습니다.") 
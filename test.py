#!/usr/bin/env python3
import sys
import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from morai_msgs.msg import CtrlCmd
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from std_msgs.msg import Float32
import time
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen

class pidController:  ## 속도 제어를 위한 PID 적용 ##
    def __init__(self, p=1.0, i=1.0, d=1.0, rate=30):
        # 비례항 : 현재 오차에 비례하여 제어 출력 생성
        self.p_gain = p 
        # 적분항 : 과거의 오차 누적에 비례하여 제어 출력 생성
        self.i_gain = i 
        # 미분항 : 오차의 변화율에 비례하여 제어 출력 생성
        self.d_gain = d 
        # controlTime : 제어 주기의 역수로 계산된 제어 시간 (제어 주기가 주어진 경우 오차의 누적 계산에 사용)
        # rate : 제어 주기 값 (제어 주기의 역수로 계산, 시간 단위로 표시)
        self.controlTime = 1/rate 
        # 이전 오차 값을 저장하는 변수, 미분항 d 를 계산하는데 사용
        self.prev_error = 0 
        # 적분 제어에 사용되는 누적된 제어 값
        self.i_control = 0 

    # target_vel : 목표 속도
    # current_ver : 현재 속도를 나타내는 벡터(이 벡터의 x성분이 현재 속도를 나타냄)
    def pid(self, target_vel, current_vel): 
        # 목표 속도와 현재 속도 사이의 오차 계산
        error = target_vel - current_vel.x 

        # 비례 제어항 : 현재 오차에 비례하여 제어 출력 계산
        p_control = self.p_gain * error 
        # 적분 제어항 : 과거 오차를 누적하여 제어 출력 계산
        self.i_control += self.i_gain * error
        # 미분 제어항 : 오차 변화율에 비례하여 제어 출력 계산
        d_control = self.d_gain * (error - self.prev_error)
        # 비례(p), 적분(i), 미분(d) 제어 항을 조합하여 최종 제어 출력 계산
        output = p_control + self.i_control + d_control
        # 이전 오차 값을 현재 오차로 업데이트
        self.prev_error = error

        return output
    # def pid에 비해 시간에 대한 의존성을 더 잘 처리함 -> 정확한 시간 변화도를 반영
    def pid_1(self, target_vel, current_vel):
        # current_vel : 벡터가 아닌 단일 값을 가정
        # def pid에서 current_vel.x과 같이 벡터의 x성분을 사용함
        # def pid_1에서 current_vel 자체가 현재 속도를 나타내는 것으로 가정
        error = target_vel - current_vel

        p_control = self.p_gain * error
        # self.controlTime(시간 간격)을 곱하여 누적 제어 값 계산
        # 적분 제어항(i)을 시간에 대한 적분으로 변환 -> 제어 시간에 대한 의존성을 줄임
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error - self.prev_error) / self.controlTime

        output = p_control + self.i_control + d_control
        self.prev_error = error
        return output

class cvThread(QThread):
    signal = pyqtSignal(QImage)

    def __init__(self, qt):
        super(cvThread, self).__init__(parent=qt)
        print('init')
        rospy.init_node('LaneDetection_Ctrl')
        self.prevTime = 0
        self.selecting_sub_image = "compressed"  # you can choose image type "compressed", "raw"
        self.isSim = True
        self.wrapCaliDone = False
        # self.steering_angle_deg = 0
        self.current_steering = 0
        self.target_steering = 0
        self.smooth_factor = 0.2
        # self.publish_rate = 10

        self.ctrl_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.pid = pidController(p=9, i=0.1, d=2.0, rate= 30) #rate = 30

        # self.steering_sub = rospy.Subscriber('/steering_angle', Int32, self.steering_callback, queue_size=1)
        self.steering_sub = rospy.Subscriber('/steering_angle', Float32, self.steering_callback, queue_size=1)

        if self.selecting_sub_image == "compressed":
            if self.isSim :
               self._sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback, queue_size=1)
               # self._sub = rospy.Subscriber('/refine_image', CompressedImage, self.callback, queue_size=1)
            else:
                self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
                self._sub = rospy.Subscriber('/zed2/zed_node/left/image_rect_color/compressed', CompressedImage, self.callback, queue_size=1)
                


        else:
            self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)

        self.bridge = CvBridge()
        self.lane = LaneDet(max_counter=3, cv_thread=self)
        
        print('init finished')

    # 차량의 카메라 또는 레이더 시스템을 정확하게 보정하는 프로세스
    def doWrapCalibration(self, cv_image):
        leftupper = (0, 0)
        rightupper = (0, 0)
        leftlower = (0, 0)
        rightlower = (0, 0)
        warped_leftupper = (0, 0)
        warped_rightupper = (0, 0)
        warped_leftlower = (0, 0)
        warped_rightlower = (0, 0)


        if self.isSim:
            # offset = 250
            # leftupper = (240, 255)
            # rightupper = (380, 255)
            # leftlower = (-380  , cv_image.shape[0])
            # rightlower = (950, cv_image.shape[0])

            # for sim
            # offset = 250
            # leftupper = (270, 250)
            # rightupper = (350, 250)
            # leftlower = (-240  , cv_image.shape[0])
            # rightlower = (880, cv_image.shape[0])
            offset = 250
            leftupper = (270, 250)
            rightupper = (350, 250)
            leftlower = (-300  , cv_image.shape[0])
            rightlower = (940, cv_image.shape[0])
            # offset = 250
            # leftupper = (300, 180)
            # rightupper = (355, 180)
            # leftlower = (0  , cv_image.shape[0])
            # rightlower = (650, cv_image.shape[0])

            warped_leftupper = (offset, 0)
            warped_rightupper = (offset, cv_image.shape[0])
            warped_leftlower = (cv_image.shape[1] - offset, 0)
            warped_rightlower = (cv_image.shape[1] - offset, cv_image.shape[0])
        else:
            offset = 250
            leftupper = (265, 160)
            rightupper = (395, 160)
            leftlower = (-135, cv_image.shape[0])
            rightlower = (795, cv_image.shape[0])

            warped_leftupper = (offset, 0)
            warped_rightupper = (offset, cv_image.shape[0])
            warped_leftlower = (cv_image.shape[1] - offset, 0)
            warped_rightlower = (cv_image.shape[1] - offset, cv_image.shape[0])


        return (np.float32([leftupper, leftlower, rightupper, rightlower]), np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower]))

    def steering_callback(self, msg):
        self.current_steering = msg.data

    
    def callback(self, image_msg):
        # try:
        if self.selecting_sub_image == "compressed":
            # converting compressed image to opencv image
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif self.selecting_sub_image == "raw":
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        if self.wrapCaliDone is False:
            wrap_origin, wrap = self.doWrapCalibration(cv_image)
            self.lane.set_presp_indices(wrap_origin, wrap)
            self.wrapCaliDone = True

        # b_out = cuv_img #cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        out_img, output_data = self.lane.process_image(cv_image)

        self.control_unit(output_data)

        h, w, ch = out_img.shape
        # size = cv_image.nbytes
        curTime = time.time()
        sec = curTime - self.prevTime
   
        self.prevTime = curTime
        fps = 1 / (sec)
        tstamp = float(image_msg.header.stamp.to_sec())
        #print(tstamp.to_sec(), type(tstamp))
        str = "FPS : %0.1f %0.3f" % (fps, tstamp)
        cv2.putText(out_img, str, (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        convertToQtFormat = QImage(out_img.data, w, h, out_img.strides[0], QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.signal.emit(p)
        
        #cv2.imshow('cv_gray', cv_image), cv2.waitKey(1)
        # except Exception as e:
        #     print(e)

    def run(self):
        rospy.spin()

    def control_unit(self, data):
        center_dist = data[0]
        left_curv = data[1]
        right_curv = data[2]

        # self.current_steering = self.current_steering + self.smooth_factor * (self.target_steering - self.current_steering)
        steering = 0#self.steering_angle_deg
        
        if center_dist < 0 :
            steering = self.steering_angle_deg
        else:
            steering = self.steering_angle_deg

        #steering = 5
        # print("steering: %.2f, center_dist : %.2f, left_curv : %.2f, right_curv : %.2f"%(steering, center_dist, left_curv, right_curv))
        # print("steering : %.2f" % (steering))
        # print("center_dist : %.2f" % (center_dist))
        # print("left_curv : %.2f" % (left_curv))
        # print("right_curv : %.2f" % (right_curv))
        # print("=====================")

        # CtrlCmd()의 경우 steering_angle의 input 값 범위가
        # -0.5 ~ 0.5 에 해당함

        # steer * 0.5 / 24
        # 0에 대한 예외 처리 필요

        if self.current_steering == 0:
            self.current_steering = 0
        else:
            self.current_steering = (self.current_steering * 0.097) / 24

        send = CtrlCmd()
        send.velocity = 1.5 #km/h
        send.accel = 0.521
        send.steering = self.current_steering#steering
        self.ctrl_pub.publish(send)

class LaneDet():
    def __init__(self, max_counter, cv_thread): #초기값 설정..?인가
        self.current_fit_left = None
        self.best_fit_left = None
        self.history_left = [np.array([False])]
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right = [np.array([False])]
        self.counter = 0
        self.max_counter = 0
        self.src = None
        self.dst = None
        self.steering_angle_deg = 0
        self.cv_thread = cv_thread

    def set_presp_indices(self, src, dest):
        self.src = src
        self.dst = dest

    def reset(self):
        self.current_fit_left = None
        self.best_fit_left = None
        self.history_left = [np.array([False])]
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right = [np.array([False])]
        self.counter = 0

    # 이전 프레임에서 얻은 양쪽 차선에 대한 다항식 피팅 결과를 사용하여 새로운 프레임에서의 차선을 추정
    def update_fit(self, left_fit, right_fit):
        if self.counter > self.max_counter: #counter가 최댓값보다 크면 0으로 reset
            self.reset()
        else:
            self.current_fit_left = left_fit
            self.current_fit_right = right_fit
            self.history_left.append(left_fit)
            self.history_right.append(right_fit)
            self.history_left = self.history_left[-self.max_counter:] if len(
                self.history_left) > self.max_counter else self.history_left
            self.history_right = self.history_right[-self.max_counter:] if len(
                self.history_right) > self.max_counter else self.history_right
            self.best_fit_left = np.mean(self.history_left, axis=0)
            self.best_fit_right = np.mean(self.history_right, axis=0)

    def find_lane_width(self, left_fitx, right_fitx, xmtr_per_pixel):
        width_pub = rospy.Publisher("/width_flag", Bool, queue_size=1)
        # Calculate lane width in pixels
        lane_width_px = right_fitx[-1] - left_fitx[-1]


        if lane_width_px > 180:
            msg = Bool()
            msg.data = False
            width_pub.publish(msg)
            #return None, None
        else:
            msg = Bool()
            msg.data = True
            width_pub.publish(msg)

        # Calculate lane width in meters
        lane_width_m = lane_width_px * xmtr_per_pixel
        

        return lane_width_px, lane_width_m
    
    # 이미지를 전처리
    def process_image(self, image):
        # self.publish_rate = 10
        # 이미지 전처리
        img = image#self.undistort_no_read(image, objpoints, imgpoints)

        kernel_size = 5
        mag_thresh = (30, 200)
        #for simulator
        r_thresh = (170, 255) #빨강
        s_thresh = (120, 255) #채도 
        b_thresh = (200, 255) #파랑
        g_thresh = (220, 255) #초록

        # r_thresh = (170, 255) #빨강
        # s_thresh = (150, 255) #채도 
        # b_thresh = (160, 255) #파랑
        # g_thresh = (220, 255) #초록

        #for zed2
        #r_thresh = (235, 255)
        #s_thresh = (165, 255)
        #b_thresh = (160, 255)
        #g_thresh = (210, 255)

        # 흰색 차선 감지를 위한 이진화 이미지 생성
        combined_binary, combined_binary255 = self.get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                                            r_thresh=r_thresh, s_thresh=s_thresh, b_thresh=b_thresh,
                                                            g_thresh=g_thresh)
        # 원근 왜곡 보정
        if self.src is not None or self.dst is not None:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = self.transform_image(combined_binary,
                                                                                                   src=self.src,
                                                                                                   dst=self.dst)
        else:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = self.transform_image(combined_binary)

        # 원본 출력
        #get origin output
        cvwraped, cvM, cvminv, cv_out_img_orig, cv_out_warped_img = self.transform_image(image)

        # 픽셀 당 미터 단위
        xmtr_per_pixel = 6.7 / 400 #3.7 / 400#800
        ymtr_per_pixel = 30 / 480 #
        cuv_img = None

        # 차선 감지
        if True or self.best_fit_left is None and self.best_fit_right is None:
            left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, cuv_img = self.fit_polynomial(
            warped, nwindows=15, margin=30, show=False)
        else:
            left_fit, right_fit, left_lane_indices, right_lane_indices = self.search_around_poly(warped, self.best_fit_left,
                                                                                              self.best_fit_right,
                                                                                              xmtr_per_pixel,
                                                                                              ymtr_per_pixel)
            
        # 차선 픽셀 디버깅
        # To debug Find our lane pixels first
        deb_leftx, deb_lefty, deb_rightx, deb_righty, deb_left_lane_indices, deb_right_lane_indices, f_line_img \
        = self.find_lines(warped, nwindows=15, margin=30, minpix=50)

        # 디버그 출력
        self.counter += 1
        birdeye_debug_img, center_coords = self.draw_birdeye_debug(warped, left_fit, right_fit, deb_leftx, deb_lefty, deb_rightx, deb_righty,
                                                                deb_left_lane_indices, deb_right_lane_indices, unwarp_matrix)
        
        # self.display_image(birdeye_debug_img)
        
        # 차선 그리기
        lane_img = self.draw_lines(img, left_fit, right_fit, unwarp_matrix)
        out_img = self.show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

       

        # 차선 피팅 업데이트
        self.update_fit(left_fit, right_fit)


        # 원본 이미지로 흑백 변환
        cv_out_img_orig = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        # 결과 이미지 병합
        # merge to horizontal
        numpy_horizontal1 = np.concatenate((out_img, f_line_img), axis=1)
        # merge to horizontal
        numpy_horizontal2 = np.concatenate((self.draw_wrapinfo(image), birdeye_debug_img), axis=1)
        num_mergy = np.concatenate((numpy_horizontal1, numpy_horizontal2), axis=0)


        # 곡률 및 중앙 위치 계산
        center_dist, left_curv, right_curv = self.publish_data(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

        lane_width_px, lane_width_m = self.find_lane_width(left_fitx, right_fitx, xmtr_per_pixel)

        if lane_width_px is not None:
            print("lane_width_m :", int(lane_width_px), "m")
            lane_center_px = (right_fitx[-1] + left_fitx[-1]) / 2
            print(lane_center_px)
            image_center_px = image.shape[1] / 2 
            center_offset_px = image_center_px - lane_center_px
            center_offset_m = center_offset_px * xmtr_per_pixel
            print("Vehicle location : %.2f" %abs(center_offset_m))

            # y_coord = unwarp_matrix.shape[0] - 1 - 180
            # y_offset_m = y_coord * ymtr_per_pixel

            # print("Y-coordinate of the lane curvature line: %d" % y_coord)
            # print("Distance to the lane curvature line: %.2f" % y_offset_m)

            vehicle_length = 1.4 #meter
            steering_angle_rad = math.atan(center_offset_m / vehicle_length)
            steering_angle_deg = max(min(float(math.degrees(steering_angle_rad)), 24), -24)
            steering_angle_deg = (float(steering_angle_deg * 14) / 24)

            angle_pub = rospy.Publisher("/steering_angle", Float32, queue_size=1)
            # rate = rospy.Rate(10)  # 10Hz
            # while not rospy.is_shutdown():
            #     msg = Int32()
            #     msg.data = steering_angle_deg
            #     angle_pub.publish(msg)
            #     rate.sleep()
            msg = Float32()
            msg.data = steering_angle_deg
            angle_pub.publish(msg)
            # rate.sleep()

            self.steering_angle_deg = steering_angle_deg
            self.cv_thread.steering_angle_deg = steering_angle_deg
            print("steering_angle_deg :", steering_angle_deg) # +는 우회전, -는 좌회전
            print("==================================")
            # self.rate = rospy.Rate(self.publish_rate)
            # self.rate.sleep()

        # lane_width_m = lane_width_px * xmtr_per_pixel
        # print(right_fitx[0], left_fitx[0])
        # lane_center_px = (right_fitx[-1] + left_fitx[-1]) / 2
        # lane_width_px = right_fitx[-1] - left_fitx[-1]
        # print("lane_width_m :", lane_width_px)
        # print("lane_center_px :", lane_center_px) # 차선 중앙
        # image_center_px = image.shape[1] / 2 
        # print("image_center_px :",image_center_px) # 이미지의 중앙값 계산
        # center_offset_px = image_center_px - lane_center_px
        # center_offset_m = center_offset_px * xmtr_per_pixel
        # print("center_offser_m :", center_offset_m)
        # L = 1.4 #meter
        # steering_angle_rad = math.atan(center_offset_m / L)
        # steering_angle_deg = max(min(int(math.degrees(steering_angle_rad)), 24), -24)

        return num_mergy, (center_dist, left_curv, right_curv, steering_angle_deg)#out_img

    def publish_data(self, img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
        (left_curvature, right_curvature) = self.radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
        center_dist, dist_txt = self.dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
        self.getTargetPoint(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
        
        # print(left_curvature, right_curvature)
        
        return center_dist, left_curvature, right_curvature 

    def draw_wrapinfo(self, img):
        out_img = np.copy(img)
        if self.src is not None and self.dst is not None:
            color_b = [0, 0, 255]
            color_g = [0, 255, 0]
            line_width = 5
            src = tuple(map(tuple, self.src))
            cv2.line(out_img, src[1], src[0], color_b, line_width)
            cv2.line(out_img, src[1], src[3], color_b, line_width * 2)
            cv2.line(out_img, src[2], src[3], color_b, line_width)
            cv2.line(out_img, src[2], src[0], color_g, line_width)

        return out_img

    def draw_birdeye_debug(self, unwarp_img, left_fit, right_fit, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, minv):
        # 선언 및 초기화
        ploty = np.linspace(0, unwarp_img.shape[0] - 1, unwarp_img.shape[0])
        unwarp_img = np.where(unwarp_img == 1, 255, unwarp_img)
        unwarp_img = cv2.cvtColor(unwarp_img, cv2.COLOR_GRAY2BGR)

        # 차선 색상 표기
        unwarp_img[lefty, leftx] = [255, 0, 0]
        unwarp_img[righty, rightx] = [0, 0, 255]

        # 중앙 차선 곡률 계산
        left_poly = np.poly1d(left_fit)
        right_poly = np.poly1d(right_fit)
        center_pol = (left_poly + right_poly) / 2
        center_fitx = center_pol(ploty)

        # 차선 포인트 시각화
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        center_coords = []
        for y, data in enumerate(left_fitx):
            left_val = unwarp_img.shape[1] - 1 if int(left_fitx[y]) >= unwarp_img.shape[1] else int(left_fitx[y])
            left_val = 0 if left_val < 0 else left_val
            right_val = unwarp_img.shape[1] - 1 if int(right_fitx[y]) >= unwarp_img.shape[1] else int(right_fitx[y])
            right_val = 0 if right_val < 0 else right_val
            # print(left_val, right_val)
            unwarp_img[y, left_val] = [255, 234, 0]
            unwarp_img[y, right_val] = [255, 234, 0]

            center_val = unwarp_img.shape[1] - 1 if int(center_fitx[y]) >= unwarp_img.shape[1] else int(center_fitx[y])
            center_val = 0 if center_val < 0 else center_val
            unwarp_img[y, center_val] = [0, 255, 0]

            if y == unwarp_img.shape[0] // 2:
                center_coords = [center_val, y]

        print(center_coords)

        # qImg = QImage(unwarp_img.data, unwarp_img.shape[1], unwarp_img.shape[0], QImage.Format_RGB888)
        # pixmap = QPixmap.fromImage(qImg)

        # painter = QPainter(pixmap)
        # pen = QPen(Qt.red)
        # painter.setPen(pen)
        # painter.drawPoint(center_coords[0], center_coords[1])
        # painter.end()

        # unwarp_img = cv2.cvtColor(np.array(pixmap.toImage()), cv2.COLOR_RGB2BGR)
        return unwarp_img, center_coords

    def get_rgb_thresh_img(self, img, channel='R', thresh=(0, 255)):
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if channel == 'R':
            bin_img = img1[:, :, 0]
        if channel == 'G':
            bin_img = img1[:, :, 1]
        if channel == 'B':
            bin_img = img1[:, :, 2]

        binary_img = np.zeros_like(bin_img).astype(np.uint8)
        binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1

        return binary_img

    def get_lab_bthresh_img(self, img, thresh=(0, 255)):
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        B = lab_img[:, :, 2]

        bin_op = np.zeros_like(B).astype(np.uint8)
        bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1

        return bin_op

    def get_lab_athresh_img(self, img, thresh=(0, 255)):
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        A = lab_img[:, :, 1]

        bin_op = np.zeros_like(A).astype(np.uint8)
        bin_op[(A >= thresh[0]) & (A < thresh[1])] = 1

        return bin_op

    def get_hls_sthresh_img(self, img, thresh=(0, 255)):
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        S = hls_img[:, :, 2]

        binary_output = np.zeros_like(S).astype(np.uint8)
        binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1

        return binary_output

    def get_bin_img(self, img, kernel_size=10, sobel_dirn='X', sobel_thresh=(0, 255), 
                    r_thresh=(0, 255), s_thresh=(0, 255), b_thresh=(0, 255), g_thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if sobel_dirn == 'X':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

        combined = np.zeros_like(sbinary)
        combined[(sbinary == 1)] = 1

        # Threshold R color channel
        r_binary = self.get_rgb_thresh_img(img, thresh=r_thresh)

        # Threshhold G color channel
        g_binary = self.get_rgb_thresh_img(img, thresh=g_thresh, channel='G')

        # Threshhold B in LAB
        b_binary = self.get_lab_bthresh_img(img, thresh=b_thresh)

        # Threshold color channel
        s_binary = self.get_hls_sthresh_img(img, thresh=s_thresh)

        # If two of the three are activated, activate in the binary image
        combined_binary = np.zeros_like(combined)
        combined_binary255 = np.zeros_like(combined)
        combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 1
        combined_binary255[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 255
        #combined_binary255[(s_binary == 1)] = 255

        return combined_binary, combined_binary255
        # return gray

    def transform_image(self, img, offset=250, src=None, dst=None):
        img_size = (img.shape[1], img.shape[0])

        out_img_orig = np.copy(img)

        leftupper = (265, 200)
        rightupper = (395, 200)
        leftlower = (-135, img.shape[0])
        rightlower = (795, img.shape[0])

        warped_leftupper = (offset, 0)
        warped_rightupper = (offset, img.shape[0])
        warped_leftlower = (img.shape[1] - offset, 0)
        warped_rightlower = (img.shape[1] - offset, img.shape[0])

        color_b = [0, 0, 255]
        color_g = [0, 255, 0]
        line_width = 10

        if src is not None:
            src = src
        else:
            src = np.float32([leftupper, leftlower, rightupper, rightlower])

        if dst is not None:
            dst = dst
        else:
            dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

        cv2.line(out_img_orig, leftlower, leftupper, color_b, line_width)
        cv2.line(out_img_orig, leftlower, rightlower, color_b, line_width * 2)
        cv2.line(out_img_orig, rightupper, rightlower, color_b, line_width)
        cv2.line(out_img_orig, rightupper, leftupper, color_g, line_width)

        # calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        minv = cv2.getPerspectiveTransform(dst, src)

        # Warp the image
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
        out_warped_img = np.copy(warped)

        cv2.line(out_warped_img, warped_rightupper, warped_leftupper, color_b, line_width)
        cv2.line(out_warped_img, warped_rightupper, warped_rightlower, color_b, line_width * 2)
        cv2.line(out_warped_img, warped_leftlower, warped_rightlower, color_b, line_width)
        cv2.line(out_warped_img, warped_leftlower, warped_leftupper, color_g, line_width)

        return warped, M, minv, out_img_orig, out_warped_img

    def find_lines(self, warped_img, nwindows=9, margin=80, minpix=40):

        # Take a histogram of the bottom half of the image
        # 영상의 아래쪽 절반을 히스토그램으로 표시합니다
        histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)

        # Create an output image to draw on and visualize the result
        # 그릴 출력 이미지를 만들고 결과를 시각화합니다
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        # 히스토그램의 왼쪽과 오른쪽 절반의 피크를 찾습니다
        # 이것들은 왼쪽과 오른쪽 라인의 시작점이 될 것입니다
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    

        # Set height of windows - based on nwindows above and image shape
        # 창 높이 설정 - 위의 n개의 창과 이미지 모양을 기준으로 합니다
        window_height = np.int(warped_img.shape[0] // nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        # 영상에서 0이 아닌 모든 픽셀의 x 및 y 위치 식별
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        

        # Current positions to be updated later for each window in nwindows
        # 각 창에 대해 나중에 업데이트할 현재 위치(nwindow)
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        # 왼쪽 및 오른쪽 레인 픽셀 인덱스를 수신할 빈 목록 만들기
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        # windows를 하나씩 통과합니다
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            # x 및 y(및 오른쪽 및 왼쪽)의 창 경계 식별
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height

            ### Find the four below boundaries of the window ###
            ### 창의 아래 4개의 경계를 찾습니다 ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

    def fit_polynomial(self, binary_warped, nwindows=5, margin=50, minpix=50, show=True):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img \
            = self.find_lines(binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

        left_fit = np.array([0, 0, 0]) if lefty.size == 0 and leftx.size == 0 else np.polyfit(lefty, leftx, 2)
        right_fit = np.array([0, 0, 0]) if rightx.size == 0 and righty.size == 0 else np.polyfit(righty, rightx, 2)


        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 255, 0]

        for y, data in enumerate(left_fitx):
            left_val = out_img.shape[1]-1 if int(left_fitx[y]) >= out_img.shape[1] else int(left_fitx[y])
            left_val = 0 if left_val < 0 else left_val
            right_val = out_img.shape[1]-1 if int(right_fitx[y]) >= out_img.shape[1] else int(right_fitx[y])
            right_val = 0 if right_val < 0 else right_val
            #print(left_val, right_val)
            out_img[y, left_val] = [255, 234, 0]
            out_img[y, right_val] = [255, 234, 0]
        # Plots the left and right polynomials on the lane lines
        # if show == True:
        #     plt.plot(left_fitx, ploty, color='yellow')
        #     plt.plot(right_fitx, ploty, color='yellow')

        return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img

    def search_around_poly(self, binary_warped, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel, margin=80):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit second order polynomial to for for points on real world
        left_lane_indices = np.polyfit(lefty * ymtr_per_pixel, leftx * xmtr_per_pixel, 2)
        right_lane_indices = np.polyfit(righty * ymtr_per_pixel, rightx * xmtr_per_pixel, 2)

        return left_fit, right_fit, left_lane_indices, right_lane_indices

    def radius_curvature(self, img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * ymtr_per_pixel, left_fitx * xmtr_per_pixel, 2)
        right_fit_cr = np.polyfit(ploty * ymtr_per_pixel, right_fitx * xmtr_per_pixel, 2)

        # find radii of curvature
        left_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ymtr_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_rad = ((1 + (2 * right_fit_cr[0] * y_eval * ymtr_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # print("Left_rad :", left_rad)
        # print("Right_rad :", right_rad)
        return (left_rad, right_rad)

    def dist_from_center(self, img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
        ## Image mid horizontal position
        # xmax = img.shape[1]*xmtr_per_pixel
        ymax = img.shape[0]# * ymtr_per_pixel

        center = img.shape[1] / 2

        lineLeft = left_fit[0] * ymax ** 2 + left_fit[1] * ymax + left_fit[2]
        lineRight = right_fit[0] * ymax ** 2 + right_fit[1] * ymax + right_fit[2]
        #print(lineRight)
        #print(right_fit)

        mid = lineLeft + (lineRight - lineLeft) / 2
        dist = (mid - center) * xmtr_per_pixel
        # print(left_fit[0])
        # print(ymax)
        # print("lineLeft :", lineLeft)
        # print("lineRight :", lineRight)
        
        # print("mid : ", mid)
        # print("lineLeft :", lineLeft) #160.2490...의 의미 : 이미지 가장 아래쪽에서 왼쪽 차선이 오른쪽으로 160.25m 위치에 있음을 의미
        # print("lineRight :", lineRight)
        # print("dist :",dist) # -0.1492.. 의 값의 의미는 차량이 중앙 위치의 왼쪽에서 약 0.15m 만큼 떨어져 있음을 의미
        # print("========================================")
        if dist >= 0.:
            message = 'Vehicle location: {:.2f} m right'.format(dist)
        else:
            message = 'Vehicle location: {:.2f} m left'.format(abs(dist))

        # print(dist)

        return dist, message

    def draw_lines(self, img, left_fit, right_fit, minv):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Find left and right points.
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

        # Warp the blank back to original image space using inverse perspective matrix
        unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]),
                                         flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

        return cv2.addWeighted(img, 1, unwarp_img, 0.3, 0)

    def show_curvatures(self, img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
        (left_curvature, right_curvature) = self.radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
        center_dist, dist_txt = self.dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)

        out_img = np.copy(img)
        avg_rad = round(np.mean([left_curvature, right_curvature]), 0)
        cv2.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return out_img

    def getTargetPoint(self, img, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        # get center curvature
        left_poly = np.poly1d(left_fit)
        right_poly = np.poly1d(right_fit)
        center_poly = (left_poly + right_poly) / 2
        center_fitx = center_poly(ploty)
        # print(left_poly, right_poly)
        a = center_poly[0]
        b = center_poly[1]
        c = center_poly[2]

        left_fit_cr = np.polyfit(ploty * ymtr_per_pixel, center_fitx * xmtr_per_pixel, 2)
        right_fit_cr = np.polyfit(ploty * ymtr_per_pixel, center_fitx * xmtr_per_pixel, 2)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 2575
        self.top = 800
        self.width = 640
        self.height = 480
        self.wsize = 640
        self.hsize = 480
        # self.label = QLabel(self)
        # self.setCentralWidget(self.label)
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        # qImg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        # pixmap = QPixmap.fromImage(qImg)
        # self.label.setPixmap(pixmap)
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(self.wsize, self.hsize)
        # create a label
        self.label = QLabel(self)
        self.label.resize(self.wsize, self.hsize)

        self.show()
        time.sleep(1)
        th = cvThread(self)
        th.signal.connect(self.setImage)
        # th.changePixmap.connect(self.setImage)
        th.start()

if __name__ == '__main__':
    # rospy.init_node("width_pub")
    # rospy.Subscriber('/')
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
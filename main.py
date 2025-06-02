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
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

# 분리한 차선 검출 알고리즘 임포트
from lane_detection import LaneDetection

class pidController:
    def __init__(self, p=1.0, i=1.0, d=1.0, rate=30):
        self.p_gain = p
        self.i_gain = i
        self.d_gain = d
        self.controlTime = 1/rate
        self.prev_error = 0
        self.i_control = 0

    def pid(self, target_vel, current_vel):
        error = target_vel - current_vel.x

        p_control = self.p_gain * error
        self.i_control += self.i_gain * error
        d_control = self.d_gain * (error - self.prev_error)
        
        output = p_control + self.i_control + d_control
        self.prev_error = error

        return output

    def pid_1(self, target_vel, current_vel):
        error = target_vel - current_vel

        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error - self.prev_error) / self.controlTime

        output = p_control + self.i_control + d_control
        self.prev_error = error
        return output

class LaneDetectionThread(QThread):
    signal = pyqtSignal(QImage)

    def __init__(self, qt):
        super(LaneDetectionThread, self).__init__(parent=qt)
        print('초기화 시작')
        
        # ROS 노드 초기화
        rospy.init_node('LaneDetection_Ctrl')
        
        # 변수 초기화
        self.prevTime = 0
        self.selecting_sub_image = "compressed"  # compressed 또는 raw
        self.isSim = True
        self.wrapCaliDone = False
        self.current_steering = 0
        self.steering_angle_deg = 0
        
        # ROS 퍼블리셔 설정
        self.ctrl_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.angle_pub = rospy.Publisher("/steering_angle", Float32, queue_size=1)
        self.width_pub = rospy.Publisher("/width_flag", Bool, queue_size=1)
        
        # PID 컨트롤러 초기화
        self.pid = pidController(p=9, i=0.1, d=2.0, rate=30)
        
        # 조향각 구독
        self.steering_sub = rospy.Subscriber('/steering_angle', Float32, self.steering_callback, queue_size=1)
        
        # 이미지 구독 설정
        if self.selecting_sub_image == "compressed":
            if self.isSim:
                self._sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback, queue_size=1)
            else:
                self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
                self._sub = rospy.Subscriber('/zed2/zed_node/left/image_rect_color/compressed', CompressedImage, self.callback, queue_size=1)
        else:
            self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)

        # 브릿지 및 차선 검출기 초기화
        self.bridge = CvBridge()
        self.lane_detector = LaneDetection()
        
        print('초기화 완료')

    def doWrapCalibration(self, cv_image):
        """원근 변환을 위한 캘리브레이션 수행"""
        if self.isSim:
            offset = 250
            leftupper = (270, 250)
            rightupper = (350, 250)
            leftlower = (-300, cv_image.shape[0])
            rightlower = (940, cv_image.shape[0])

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

        return (np.float32([leftupper, leftlower, rightupper, rightlower]),
                np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower]))

    def steering_callback(self, msg):
        """조향각 콜백 함수"""
        self.current_steering = msg.data

    def callback(self, image_msg):
        """이미지 콜백 함수"""
        try:
            # 이미지 변환
            if self.selecting_sub_image == "compressed":
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            elif self.selecting_sub_image == "raw":
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # 원근 변환 설정
            if self.wrapCaliDone is False:
                wrap_origin, wrap = self.doWrapCalibration(cv_image)
                self.lane_detector.set_perspective_transform(wrap_origin, wrap)
                self.wrapCaliDone = True

            # 차선 검출 처리
            result = self.lane_detector.process_image(cv_image)
            
            # 차선 정보 추출
            left_fit = result['left_fit']
            right_fit = result['right_fit']
            left_fitx = result['left_fitx']
            right_fitx = result['right_fitx']
            lane_img = result['lane_image']
            
            # 픽셀 당 미터 단위 설정
            xmtr_per_pixel = 6.7 / 400
            
            # 차량 위치 및 조향각 계산
            center_dist, steering_angle_deg = self.calculate_vehicle_position(
                cv_image, left_fit, right_fit, left_fitx, right_fitx, xmtr_per_pixel
            )
            
            # 조향각 설정 및 발행
            if steering_angle_deg is not None:
                self.steering_angle_deg = steering_angle_deg
                msg = Float32()
                msg.data = steering_angle_deg
                self.angle_pub.publish(msg)
                print(f"조향각: {steering_angle_deg:.2f}")
                print("==================================")

            # 차량 제어
            self.control_unit(center_dist)

            # FPS 계산
            h, w, ch = lane_img.shape
            curTime = time.time()
            sec = curTime - self.prevTime
            self.prevTime = curTime
            fps = 1 / (sec)
            tstamp = float(image_msg.header.stamp.to_sec())
            
            # FPS 표시
            cv2.putText(lane_img, f"FPS: {fps:.1f} {tstamp:.3f}", (w-300, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            # 이미지 변환 및 표시
            convertToQtFormat = QImage(lane_img.data, w, h, lane_img.strides[0], QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.signal.emit(p)
            
        except Exception as e:
            print(f"오류 발생: {e}")

    def run(self):
        """스레드 실행 함수"""
        rospy.spin()

    def calculate_vehicle_position(self, image, left_fit, right_fit, left_fitx, right_fitx, xmtr_per_pixel):
        """차량 위치와 조향각 계산"""
        # 차선 폭 계산
        lane_width_px = right_fitx[-1] - left_fitx[-1]
        
        # 차선 폭이 너무 넓으면 무시
        if lane_width_px > 180:
            msg = Bool()
            msg.data = False
            self.width_pub.publish(msg)
            return 0, None
        else:
            msg = Bool()
            msg.data = True
            self.width_pub.publish(msg)
        
        # 차선 중앙과 이미지 중앙 간의 오프셋 계산
        lane_center_px = (right_fitx[-1] + left_fitx[-1]) / 2
        image_center_px = image.shape[1] / 2
        center_offset_px = image_center_px - lane_center_px
        center_offset_m = center_offset_px * xmtr_per_pixel
        
        print(f"차량 위치: {abs(center_offset_m):.2f} m")
        
        # 조향각 계산
        vehicle_length = 1.4  # meter
        steering_angle_rad = math.atan(center_offset_m / vehicle_length)
        steering_angle_deg = max(min(float(math.degrees(steering_angle_rad)), 24), -24)
        steering_angle_deg = (float(steering_angle_deg * 14) / 24)
        
        return center_offset_m, steering_angle_deg

    def control_unit(self, center_dist):
        """차량 제어 명령 생성"""
        # 조향각 변환
        if self.current_steering == 0:
            self.current_steering = 0
        else:
            self.current_steering = (self.current_steering * 0.097) / 24

        # 제어 명령 생성 및 발행
        send = CtrlCmd()
        send.velocity = 1.5  # km/h
        send.accel = 0.521
        send.steering = self.current_steering
        self.ctrl_pub.publish(send)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = '차선 검출 및 자율주행'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        """이미지 업데이트 함수"""
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        """UI 초기화"""
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        # 라벨 생성
        self.label = QLabel(self)
        self.label.resize(self.width, self.height)
        self.setCentralWidget(self.label)

        self.show()
        
        # 스레드 시작
        self.thread = LaneDetectionThread(self)
        self.thread.signal.connect(self.setImage)
        self.thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_()) 
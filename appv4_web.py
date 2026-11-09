#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HỆ THỐNG PHÂN LOẠI CÀ CHUA - GIAO DIỆN WEB DASHBOARD CAO CẤP
Kế thừa và bảo toàn 100% thông số & logic phân loại của appv3.py & firmware maiiV3.ino.
"""

import os
import glob
import time
import json
import queue
import asyncio
import datetime
import threading
import webbrowser
import unicodedata
from functools import lru_cache
from contextlib import asynccontextmanager

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import numpy as np
except ImportError:
    np = None

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, StreamingResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Route, Mount, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

# ==========================================================
# THÔNG SỐ CỐ ĐỊNH & LOGIC CỐT LÕI (GIỮ NGUYÊN TỪ APPV3.PY)
# ==========================================================
MODEL_DIR = r"C:\Yolo v11\model"
CAM_W, CAM_H = 640, 480

ROI_W = 400
ROI_H = 450
ROI_X1 = (CAM_W - ROI_W) // 2
ROI_Y1 = (CAM_H - ROI_H) // 2
ROI_X2 = ROI_X1 + ROI_W
ROI_Y2 = ROI_Y1 + ROI_H

CONSECUTIVE_CLASS_FRAMES_TO_FINALIZE = 20
RESULT_DISPLAY_MS = 5000

FALLBACK_CLASSES = {
    0: "Hỏng",
    1: "Chín",
    2: "Ương",
    3: "Xanh",
}

CLASS_COLORS = {
    0: (255, 0, 0),      # BGR: blue - Hỏng
    1: (0, 0, 255),      # red - Chín
    2: (0, 165, 255),    # orange - Ương
    3: (0, 255, 0),      # green - Xanh
}

@lru_cache(maxsize=128)
def normalize_label(label):
    """Chuẩn hóa nhãn tiếng Việt từ YOLO."""
    text = str(label).strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return text.replace("đ", "d")

MODEL_LABEL_TO_SYSTEM_ID = {
    "hong": 0,
    "chin": 1,
    "uong": 2,
    "xanh": 3,
    "tomato hong": 0,
    "tomato chin": 1,
    "tomato uong": 2,
    "tomato xanh": 3,
}

def model_label_to_system_id(label):
    return MODEL_LABEL_TO_SYSTEM_ID.get(normalize_label(label))

def find_model():
    candidates = [
        os.path.join(MODEL_DIR, "best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_small_cachua_4class", "yolo11s_tomato", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_small_cachua_4class", "yolo11s_tomato", "weights", "best.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    found = glob.glob(os.path.join(MODEL_DIR, "**", "best.pt"), recursive=True)
    if found:
        return found[0]
    local_found = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "**", "best.pt"), recursive=True)
    return local_found[0] if local_found else None

def bbox_touches_roi(x1, y1, x2, y2):
    """Bounding box giao với ROI 400x450."""
    return not (x2 < ROI_X1 or x1 > ROI_X2 or y2 < ROI_Y1 or y1 > ROI_Y2)


# ==========================================================
# QUẢN LÝ HỆ THỐNG PHÂN LOẠI (BACKEND ENGINE)
# ==========================================================
class TomatoSorterEngine:
    def __init__(self):
        self.cap = None
        self.model = None
        self.ser = None
        self.running = False
        self.camera_index = 0
        self.com_port = ""
        self.fps = 0.0
        self.last_frame_time = None
        self.latest_encoded_frame = None

        # Lock & luồng
        self._serial_lock = threading.Lock()
        self._serial_buffer = ""
        self._state_lock = threading.Lock()
        self._inference_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._inference_thread = None
        self._capture_thread = None
        self._serial_poll_thread = None

        # GPU / CUDA detection
        self._device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self._device = 0
        except Exception:
            self._device = "cpu"

        # ---------------- CẤU HÌNH AI & PHÂN LOẠI ----------------
        self.conf_th = 0.40
        self.iou_th = 0.45
        self.imgsz = "480"
        self.type12_th = 120.0
        self.type3_min_th = 60.0
        self.scale_factor = 930.0
        self.speed = 100
        self.servo_ms = 3000
        self.show_roi = False
        self.active_preset = "vua"

        # Telemetry ngoại vi thời gian thực
        self.tof1_dist = None
        self.tof2_dist = None
        self.feeder_s3_angle = 0
        self.feeder_s4_angle = 0
        self.servo1_angle = 90
        self.servo2_angle = 270

        # Đo thời gian phiên làm việc
        self.session_start_time = None
        self.accumulated_runtime = 0.0

        # ---------------- BỘ NHỚ QUẢ & TIẾN TRÌNH ----------------
        self.detect_latched = False
        self.latched_class = -1
        self.latched_conf = 0.0
        self.consecutive_class = None
        self.consecutive_frames = 0
        self.consecutive_confidence_sum = 0.0
        self.vote_finalized = False
        self.last_max_weight = None
        self.last_finalized_weight = None
        self.last_result = "--"
        self.result_active = False
        self.active_result_type = None
        self.result_clear_timer = None

        # Tín hiệu đèn cho 8 bước (True = sáng đèn)
        self.step_signals = {
            "1": False,
            "2": False,
            "3": False,
            "4": False,
            "5": False,
            "6": False,
            "7A": False,
            "7B": False,
        }

        # Trạng thái văn bản
        self.status_text = "Sẵn sàng"
        self.ai_status_text = "AI: chưa tải model"
        self.esp_status_text = "ESP32: chưa kết nối"
        self.cycle_status_text = "Chu kỳ: chờ quả"
        self.tare_status_text = "Tare: chưa chạy"
        self.tare_state = "idle"
        self.vl53_1_status_text = "Cảm biến 1: chưa có số đo (SDA4)"
        self.vl53_2_status_text = "Cảm biến 2: chưa có số đo (SDA21)"
        self.feeder_status_text = "Cụm cấp (S3+S4): chờ TARE hoàn tất"

        # Detections trong frame
        self.current_detections = []

        # Thống kê tổng số lượng & khối lượng
        self.stats = {
            "type1_weight": 0.0,
            "type1_count": 0,
            "type2_weight": 0.0,
            "type2_count": 0,
            "type3_weight": 0.0,
            "type3_count": 0,
            "reject_count": 0,
            "reject_weight": 0.0,
        }

        # Lịch sử phân loại
        self.history = []

        # Danh sách WebSocket kết nối
        self.active_websockets = set()

        # Tự động tìm COM ban đầu
        ports = self.get_available_ports()
        if ports:
            self.com_port = ports[0]

        # Tải cấu hình đã lưu
        self.load_config_file()

    def load_config_file(self):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.conf_th = float(cfg.get("conf", self.conf_th))
                self.iou_th = float(cfg.get("iou", self.iou_th))
                self.imgsz = str(cfg.get("imgsz", self.imgsz))
                self.type12_th = float(cfg.get("type12", self.type12_th))
                self.type3_min_th = float(cfg.get("type3_min", self.type3_min_th))
                self.scale_factor = float(cfg.get("scale", self.scale_factor))
                self.speed = int(cfg.get("speed", self.speed))
                self.servo_ms = int(cfg.get("servo_ms", self.servo_ms))
                self.active_preset = str(cfg.get("preset", self.active_preset))
            except Exception:
                pass

    def save_config_file(self):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            cfg = {
                "conf": self.conf_th,
                "iou": self.iou_th,
                "imgsz": self.imgsz,
                "type12": self.type12_th,
                "type3_min": self.type3_min_th,
                "scale": self.scale_factor,
                "speed": self.speed,
                "servo_ms": self.servo_ms,
                "preset": self.active_preset
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def restore_defaults(self):
        self.conf_th = 0.40
        self.iou_th = 0.45
        self.imgsz = "480"
        self.type12_th = 120.0
        self.type3_min_th = 60.0
        self.scale_factor = 930.0
        self.speed = 100
        self.servo_ms = 3000
        self.active_preset = "vua"
        self.save_config_file()
        self.send_all_config()

    # ==========================================================
    # QUẢN LÝ SERIAL ESP32
    # ==========================================================
    def get_available_ports(self):
        if list_ports is None:
            return []
        return [p.device for p in list_ports.comports()]

    def get_available_cameras(self):
        if cv2 is None:
            return ["0", "1", "2"]
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            cap.release()
            if ok:
                found.append(str(i))
        return found if found else ["0", "1"]

    def connect_serial(self, port):
        self.disconnect_serial()
        if serial is None or not port:
            return False, "Thiếu thư viện pyserial hoặc chưa chọn cổng COM."
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.05)
            self._serial_buffer = ""
            self.com_port = port
            self.esp_status_text = f"ESP32: {port} @115200"
            # Khởi chạy luồng đọc Serial liên tục
            self._serial_poll_thread = threading.Thread(target=self._serial_worker, daemon=True)
            self._serial_poll_thread.start()
            # Gửi cấu hình sau 250ms
            threading.Timer(0.25, self.send_all_config).start()
            return True, f"Đã kết nối thành công {port}"
        except Exception as exc:
            self.ser = None
            self.esp_status_text = f"ESP32: lỗi kết nối ({exc})"
            return False, str(exc)

    def disconnect_serial(self):
        if self.ser and self.ser.is_open:
            with self._serial_lock:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None
        self.esp_status_text = "ESP32: chưa kết nối"

    def send(self, text):
        if self.ser is None or not self.ser.is_open:
            return False
        with self._serial_lock:
            try:
                data = (text.strip() + "\n").encode("utf-8")
                self.ser.write(data)
                self.ser.flush()
                return True
            except Exception as exc:
                self.esp_status_text = f"ESP32: lỗi gửi ({type(exc).__name__})"
                return False

    def send_all_config(self):
        if self.ser is None or not self.ser.is_open:
            return
        payload = (
            f"CFG CONF {self.conf_th:.3f}\n"
            f"CFG SCALE {self.scale_factor:.3f}\n"
            f"CFG SPEED {self.speed}\n"
            f"CFG TYPE12 {self.type12_th:.2f}\n"
            f"CFG TYPE3 {self.type3_min_th:.2f}\n"
            f"CFG SERVO_MS {self.servo_ms}\n"
        )
        with self._serial_lock:
            try:
                self.ser.write(payload.encode("utf-8"))
                self.ser.flush()
            except Exception as exc:
                self.esp_status_text = f"ESP32: lỗi gửi ({type(exc).__name__})"

    def _serial_worker(self):
        while self.ser and self.ser.is_open:
            try:
                n = self.ser.in_waiting
                if n > 0:
                    with self._serial_lock:
                        raw = self.ser.read(n)
                    self._serial_buffer += raw.decode("utf-8", errors="ignore")
                    while "\n" in self._serial_buffer:
                        line, self._serial_buffer = self._serial_buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            self.handle_esp_line(line)
            except Exception:
                pass
            time.sleep(0.01)

    # ==========================================================
    # XỬ LÝ PHẢN HỒI SERIAL TỪ ESP32 (CHUẨN MAIIV3.INO)
    # ==========================================================
    def handle_esp_line(self, line):
        if line == "READY":
            self.esp_status_text = "ESP32: sẵn sàng"
            self.send_all_config()
            return

        if line == "VL53_OK":
            self.vl53_2_status_text = "Cảm biến 2: sẵn sàng (SDA21/SCL22)"
            return

        if line == "VL53_1_OK":
            self.vl53_1_status_text = "Cảm biến 1: sẵn sàng (SDA4/SCL5)"
            return

        if line.startswith("ERROR,VL53_1_INIT"):
            self.vl53_1_status_text = "Cảm biến 1: LỖI KHỞI TẠO — kiểm tra dây SDA4/SCL5"
            self.feeder_status_text = "Servo 3: khóa cấp quả vì Cảm biến 1 lỗi"
            self.status_text = "Lỗi cảm biến 1"
            return

        if line.startswith("VL53_1_OBJECT,1,DIST,"):
            try:
                distance = int(line.rsplit(",", 1)[1])
                self.tof1_dist = distance
                self.vl53_1_status_text = f"Cảm biến 1: CÓ QUẢ — {distance} mm"
                self.step_signals["3"] = True
            except ValueError:
                self.vl53_1_status_text = "Cảm biến 1: phát hiện quả"
                self.step_signals["3"] = True
            return

        if line.startswith("VL53_1_CLEAR,DIST,"):
            try:
                distance = int(line.rsplit(",", 1)[1])
                self.tof1_dist = distance
                self.vl53_1_status_text = f"Cảm biến 1: trống — {distance} mm"
            except ValueError:
                self.vl53_1_status_text = "Cảm biến 1: trống"
            self.step_signals["3"] = False
            return

        if line == "VL53_1_REARMED":
            self.vl53_1_status_text = "Cảm biến 1: đã mở khóa, chờ quả mới"
            self.step_signals["3"] = False
            return

        if line.startswith("VL53_2_OBJECT,1,DIST,"):
            try:
                distance = int(line.rsplit(",", 1)[1])
                self.tof2_dist = distance
                self.vl53_2_status_text = f"Cảm biến 2: PHÁT HIỆN — {distance} mm"
                self.step_signals["6"] = True
            except ValueError:
                self.vl53_2_status_text = "Cảm biến 2: phát hiện quả — chốt"
                self.step_signals["6"] = True
            return

        if line.startswith("VL53_2_CLEAR,DIST,"):
            try:
                distance = int(line.rsplit(",", 1)[1])
                self.tof2_dist = distance
                self.vl53_2_status_text = f"Cảm biến 2: trống — {distance} mm"
            except ValueError:
                self.vl53_2_status_text = "Cảm biến 2: trống"
            self.step_signals["6"] = False
            return

        if line.startswith("FEEDER_START,CAUSE,"):
            cause = line.split(",")[-1]
            cause_text = {
                "STARTUP": "cấp quả đầu sau TARE",
                "VL53_2": "cấp quả kế tiếp",
                "RETRY_10S": "thử cấp lại (10s)",
            }.get(cause, cause)
            self.feeder_status_text = f"Cụm cấp: S4 đóng chặn (0°), S3 mở xả (90°) [{cause_text}]"
            self.step_signals["2"] = True
            return

        if line == "FEEDER_HOME,WAIT_VL53_2" or line == "FEEDER_HOME":
            self.feeder_status_text = "Cụm cấp: S3 & S4 đã về vị trí mặc định (20°)"
            self.step_signals["2"] = False
            return

        if line == "FEED_TEST_DONE":
            self.feeder_status_text = "Cụm cấp: Đã test xong chu kỳ cấp 2 servo (về 20°)"
            return

        if line.startswith("FEEDER_WAIT,"):
            try:
                seconds = int(line.split(",", 1)[1]) / 1000.0
            except ValueError:
                seconds = 10.0
            self.feeder_status_text = f"Cụm cấp: chờ thử lại sau {seconds:.0f}s"
            self.step_signals["2"] = False
            return

        if line == "FEEDER_FRUIT_SEEN":
            self.feeder_status_text = "Cụm cấp: quả đã qua Cảm biến 1"
            return

        if line.startswith("CHECK,"):
            # CHECK,VL53_2_DIST,120,VL53_1_DIST,200,HX711_READY,1,HX711_RAW,12345
            self.status_text = f"Chẩn đoán: {line}"
            return

        if line == "TARE_START":
            self.cycle_status_text = "Chu kỳ: đang ổn định / tare 10s"
            self.tare_status_text = "ĐANG TARE CÂN (10s)..."
            self.tare_state = "running"
            self.step_signals["1"] = True
            return

        if line == "TARE_DONE":
            self.cycle_status_text = "Chu kỳ: sẵn sàng"
            self.tare_status_text = "✓ TARE HOÀN TẤT"
            self.tare_state = "ready"
            self.step_signals["1"] = False
            self.feeder_status_text = "Servo 3: sẵn sàng cấp quả"
            return

        if line == "STOPPED":
            self.cycle_status_text = "Chu kỳ: dừng"
            self.tare_status_text = "Hệ thống dừng"
            self.tare_state = "stopped"
            self.step_signals = {k: False for k in self.step_signals}
            self.reset_fruit_memory()
            return

        if line == "RESET":
            self.reset_fruit_memory()
            self.cycle_status_text = "Chu kỳ: chờ quả tiếp theo"
            self.step_signals = {k: False for k in self.step_signals}
            return

        if line.startswith("AI_LATCHED,"):
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    cls_id = int(parts[1])
                    conf = float(parts[2])
                    self.detect_latched = True
                    self.latched_class = cls_id
                    self.latched_conf = conf
                    self.step_signals["4"] = True
                    self.cycle_status_text = "Chu kỳ: ESP32 đã nhận lớp AI"
                except ValueError:
                    pass
            return

        if line.startswith("TRIGGER,MAX,"):
            # Ví dụ: TRIGGER,MAX,131.26,CLASS,2,DETECT,1,AI_OK,1,WEIGHT_OK,1,VL53_2,1,RESULT,TYPE3
            parts = line.split(",")
            try:
                max_idx = parts.index("MAX") + 1
                self.last_max_weight = float(parts[max_idx])
                self.last_finalized_weight = self.last_max_weight
                self.step_signals["5"] = True
            except (ValueError, IndexError):
                pass

            self.detect_latched = True
            self.vote_finalized = True

            if "RESULT" in parts:
                try:
                    result = parts[parts.index("RESULT") + 1]
                    self.last_result = result
                    self.active_result_type = result
                    self.show_result_for_five_seconds(result)

                    # Cập nhật thống kê & lịch sử
                    self.record_classification(result, self.last_max_weight, self.latched_class, self.latched_conf)
                except (ValueError, IndexError):
                    pass
            return

        if line.startswith("SERVO_WAIT,"):
            self.cycle_status_text = "Chu kỳ: Servo 2 chờ 1s trước khi gạt"
            self.step_signals["7A"] = True
            return

        if line.startswith("SERVO,"):
            self.cycle_status_text = "Chu kỳ: servo đang gạt"
            if self.last_result == "REJECT":
                self.step_signals["7A"] = True
                self.step_signals["7B"] = False
            else:
                self.step_signals["7A"] = False
                self.step_signals["7B"] = True
            return

        if line.startswith("SERVO_HOME"):
            self.step_signals["7A"] = False
            self.step_signals["7B"] = False
            return

        if line.startswith("ERROR,"):
            self.status_text = line
            return

    # ==========================================================
    # XỬ LÝ KẾT QUẢ & LỊCH SỬ PHÂN LOẠI
    # ==========================================================
    def record_classification(self, result_code, weight, class_id, conf):
        w = float(weight) if weight is not None else 0.0
        fruit_name = FALLBACK_CLASSES.get(class_id, "Không xác định")

        badge = "LB"
        status_tag = "Đạt"
        full_name = "Loại bỏ"

        # Cập nhật tổng khối lượng & số lượng
        if result_code == "TYPE1":
            self.stats["type1_count"] += 1
            self.stats["type1_weight"] += w
            servo_action = "Servo 1 (35°)"
            badge = "L1"
            full_name = "Ương to"
            status_tag = "Đạt"
        elif result_code == "TYPE2":
            self.stats["type2_count"] += 1
            self.stats["type2_weight"] += w
            servo_action = "Servo 1 (90°)"
            badge = "L2"
            full_name = "Chín to"
            status_tag = "Đạt"
        elif result_code == "TYPE3":
            self.stats["type3_count"] += 1
            self.stats["type3_weight"] += w
            servo_action = "Servo 1 (145°)"
            badge = "L3"
            full_name = "Chín nhỏ" if fruit_name == "Chín" else "Ương nhỏ"
            status_tag = "Đạt"
        else:
            self.stats["reject_count"] += 1
            if w > 0:
                self.stats["reject_weight"] += w
            servo_action = "Servo 2 (0°)"
            badge = "LB"
            full_name = "Loại bỏ"
            if w > 0 and w < self.type3_min_th:
                status_tag = f"Dưới {int(self.type3_min_th)} g"
            else:
                status_tag = fruit_name if fruit_name in ["Hỏng", "Xanh"] else "Loại bỏ"

        entry = {
            "id": len(self.history) + 1,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "full_time": datetime.datetime.now().strftime("%H:%M:%S - %d/%m/%Y"),
            "fruit": fruit_name,
            "badge": badge,
            "full_name": full_name,
            "status_tag": status_tag,
            "weight": f"{w:.1f} g",
            "weight_raw": round(w, 1),
            "conf": f"{(conf * 100):.1f}%" if conf > 0 else "--",
            "conf_raw": round(conf, 2) if conf > 0 else 0.0,
            "result": full_name,
            "result_code": result_code,
            "servo": servo_action,
        }
        self.history.append(entry)

    def show_result_for_five_seconds(self, result):
        if self.result_clear_timer:
            self.result_clear_timer.cancel()
        self.result_active = True
        self.result_clear_timer = threading.Timer(
            RESULT_DISPLAY_MS / 1000.0, self._clear_result_display
        )
        self.result_clear_timer.start()

    def _clear_result_display(self):
        self.result_active = False
        self.last_result = "--"
        self.active_result_type = None
        self.step_signals["5"] = False
        self.step_signals["7A"] = False
        self.step_signals["7B"] = False

    def reset_fruit_memory(self):
        self.detect_latched = False
        self.latched_class = -1
        self.latched_conf = 0.0
        self.consecutive_class = None
        self.consecutive_frames = 0
        self.consecutive_confidence_sum = 0.0
        self.vote_finalized = False
        self.last_max_weight = None
        self.step_signals["4"] = False
        self.step_signals["5"] = False

    # ==========================================================
    # LUỒNG CAMERA & SUY LUẬN YOLO (GPU CUDA)
    # ==========================================================
    def start(self, camera_idx):
        if self.running:
            return True, "Hệ thống đang chạy."

        if self.ser is None or not self.ser.is_open:
            return False, "ESP32 chưa kết nối! Hãy kết nối cổng COM trước khi chạy."

        if cv2 is None:
            return False, "Chưa cài đặt OpenCV (cv2)."

        camera_idx = int(camera_idx)
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            return False, f"Không thể mở Camera số {camera_idx}."

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_index = camera_idx

        # Tải mô hình YOLO
        model_path = find_model()
        if YOLO is None:
            self.cap.release()
            self.cap = None
            return False, "Chưa cài thư viện ultralytics."

        if model_path is None:
            self.cap.release()
            self.cap = None
            return False, f"Không tìm thấy best.pt trong thư mục: {MODEL_DIR}"

        try:
            self.model = YOLO(model_path)
            self.ai_status_text = f"AI: {os.path.basename(model_path)} ({self._device})"
            # Warm-up 1 frame
            if np is not None:
                dummy = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
                self.model.predict(
                    source=dummy,
                    imgsz=int(self.imgsz),
                    device=self._device,
                    verbose=False,
                )
        except Exception as exc:
            self.cap.release()
            self.cap = None
            return False, f"Lỗi tải model YOLO: {exc}"

        self.reset_fruit_memory()
        self.last_frame_time = None
        self.fps = 0.0
        self.running = True
        self.session_start_time = time.time()

        # Luồng suy luận
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        # Luồng đọc camera
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()

        self.send_all_config()
        self.send("RUN 1")
        self.status_text = f"Đang chạy | Camera {camera_idx}"
        self.cycle_status_text = "Chu kỳ: ESP32 đang tare 10s"
        self.tare_status_text = "ĐANG HIỆU CHUẨN CÂN (10s)..."
        self.tare_state = "running"
        self.active_step = "1"

        return True, "Hệ thống đã bắt đầu chạy thành công."

    def stop(self):
        self.running = False
        if self.session_start_time:
            self.accumulated_runtime += (time.time() - self.session_start_time)
            self.session_start_time = None

        if self._inference_thread and self._inference_thread.is_alive():
            try:
                self._inference_thread.join(timeout=0.2)
            except Exception:
                pass

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.send("RUN 0")
        self.status_text = "Đã dừng"
        self.cycle_status_text = "Chu kỳ: dừng"
        self.tare_status_text = "Hệ thống đã dừng"
        self.tare_state = "stopped"
        self.fps = 0.0
        self.latest_encoded_frame = None

    def _capture_worker(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]
            if w != CAM_W or h != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))

            # Gửi vào luồng suy luận nếu rảnh
            try:
                self._inference_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            # Lấy kết quả suy luận
            display_frame = frame
            inference_result = None
            try:
                display_frame, inference_result = self._result_queue.get_nowait()
            except queue.Empty:
                pass

            # Tính FPS
            now = time.perf_counter()
            if self.last_frame_time is not None:
                dt = now - self.last_frame_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.fps = 0.85 * self.fps + 0.15 * instant_fps
            self.last_frame_time = now

            # Vẽ khung ROI nếu được bật tùy chọn
            if self.show_roi:
                cv2.rectangle(
                    display_frame,
                    (ROI_X1, ROI_Y1),
                    (ROI_X2, ROI_Y2),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "VUNG ROI 400x450",
                    (ROI_X1 + 10, ROI_Y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # Vẽ FPS
            cv2.putText(
                display_frame,
                f"FPS: {self.fps:.1f}",
                (CAM_W - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # Xử lý kết quả suy luận & logic 20 frame liên tiếp
            if inference_result is not None:
                detections, candidates = inference_result
                with self._state_lock:
                    self.current_detections = detections
                if not self.detect_latched and not self.vote_finalized:
                    self.process_frame_votes(candidates)

            # Encode JPEG cho Web Stream
            ret, jpeg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                self.latest_encoded_frame = jpeg.tobytes()

            time.sleep(0.015)

    def _inference_worker(self):
        while self.running:
            try:
                frame = self._inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            detections, candidates = self.detect(frame)
            try:
                self._result_queue.put_nowait((frame, (detections, candidates)))
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._result_queue.put_nowait((frame, (detections, candidates)))

    def detect(self, frame):
        if self.model is None:
            return [], []

        detections = []
        candidates = []
        try:
            result = self.model.predict(
                source=frame,
                conf=self.conf_th,
                iou=self.iou_th,
                imgsz=int(self.imgsz),
                device=self._device,
                verbose=False,
            )[0]

            names = result.names
            if result.boxes is None:
                return detections, candidates

            for b in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])

                if isinstance(names, dict):
                    raw_label = names.get(cls_id, str(cls_id))
                else:
                    raw_label = str(names[cls_id]) if cls_id < len(names) else str(cls_id)

                system_id = model_label_to_system_id(raw_label)
                display_label = FALLBACK_CLASSES.get(system_id, str(raw_label))
                in_roi = bbox_touches_roi(x1, y1, x2, y2)

                detections.append({
                    "model_id": cls_id,
                    "id": system_id,
                    "label": display_label,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "in_roi": in_roi,
                })

                if in_roi and system_id is not None and conf > self.conf_th:
                    candidates.append((conf, system_id, len(detections) - 1))

                # Vẽ bounding box trên frame
                if in_roi:
                    draw_color = CLASS_COLORS.get(system_id, (255, 255, 255))
                    thickness = 3
                else:
                    draw_color = (110, 110, 110)
                    thickness = 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)
                cv2.putText(
                    frame,
                    display_label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    draw_color,
                    2 if in_roi else 1,
                )
        except Exception:
            pass

        return detections, candidates

    def process_frame_votes(self, candidates):
        """Logic chốt 20 frame liên tiếp theo đúng quy chuẩn appv3.py."""
        if self.detect_latched or self.vote_finalized:
            return

        if not candidates:
            self.consecutive_class = None
            self.consecutive_frames = 0
            self.consecutive_confidence_sum = 0.0
            self.step_signals["4"] = False
            return

        conf, system_id, _ = max(candidates, key=lambda item: item[0])
        if system_id not in FALLBACK_CLASSES:
            self.consecutive_class = None
            self.consecutive_frames = 0
            self.consecutive_confidence_sum = 0.0
            self.step_signals["4"] = False
            return

        self.step_signals["4"] = True

        if system_id == self.consecutive_class:
            self.consecutive_frames += 1
            self.consecutive_confidence_sum += float(conf)
        else:
            self.consecutive_class = int(system_id)
            self.consecutive_frames = 1
            self.consecutive_confidence_sum = float(conf)

        if self.consecutive_frames >= CONSECUTIVE_CLASS_FRAMES_TO_FINALIZE:
            self.finalize_consecutive_class()

    def finalize_consecutive_class(self):
        if (
            self.vote_finalized
            or self.detect_latched
            or self.consecutive_class not in FALLBACK_CLASSES
            or self.consecutive_frames < CONSECUTIVE_CLASS_FRAMES_TO_FINALIZE
        ):
            return

        winner = int(self.consecutive_class)
        avg_conf = self.consecutive_confidence_sum / self.consecutive_frames
        self.vote_finalized = True
        self.detect_latched = True
        self.latched_class = winner
        self.latched_conf = float(avg_conf)
        self.step_signals["4"] = True

        self.cycle_status_text = "Chu kỳ: đã gửi lớp AI cho ESP32"
        self.send(f"AI {winner} {avg_conf:.4f}")

    # ==========================================================
    # LẤY STATE TOÀN BỘ ĐỂ GỬI LÊN WEB
    # ==========================================================
    def get_full_state(self):
        with self._state_lock:
            avg_conf = (
                (self.consecutive_confidence_sum / self.consecutive_frames)
                if self.consecutive_frames > 0
                else 0.0
            )
            latched_label = (
                FALLBACK_CLASSES.get(self.latched_class, "chưa có")
                if self.detect_latched
                else "chưa có"
            )

            result_label_map = {
                "TYPE1": "LOẠI 1 (ƯƠNG TO)",
                "TYPE2": "LOẠI 2 (CHÍN TO)",
                "TYPE3": "LOẠI 3 (CHÍN NHỎ)",
                "REJECT": "LOẠI BỎ",
            }

            active_fruit = "--"
            if self.consecutive_class is not None:
                active_fruit = FALLBACK_CLASSES.get(self.consecutive_class, "--")
            elif self.detect_latched:
                active_fruit = FALLBACK_CLASSES.get(self.latched_class, "--")

            # Tính thời gian phiên làm việc & thống kê
            current_runtime = self.accumulated_runtime
            if self.running and self.session_start_time:
                current_runtime += (time.time() - self.session_start_time)
            
            work_hours = round(current_runtime / 3600.0, 2)
            runtime_mins = current_runtime / 60.0
            total_fruits = (
                self.stats["type1_count"] + self.stats["type2_count"] +
                self.stats["type3_count"] + self.stats["reject_count"]
            )
            total_weight_kg = round(
                (self.stats["type1_weight"] + self.stats["type2_weight"] + 
                 self.stats["type3_weight"] + self.stats.get("reject_weight", 0.0)) / 1000.0, 2
            )
            throughput = round(total_fruits / runtime_mins, 1) if runtime_mins > 0.1 else 0.0

            return {
                "running": self.running,
                "serial_connected": (self.ser is not None and self.ser.is_open),
                "camera_index": self.camera_index,
                "com_port": self.com_port,
                "fps": self.fps,
                "device": self._device if isinstance(self._device, str) else "cuda:0",
                "step_signals": self.step_signals,
                "active_result_type": self.active_result_type,
                "max_weight": self.last_max_weight if self.last_max_weight is not None else self.last_finalized_weight,
                "latched_text": f"Detect nhớ: {latched_label}",
                "active_fruit_name": active_fruit,
                "consecutive_frames": self.consecutive_frames,
                "target_consecutive": CONSECUTIVE_CLASS_FRAMES_TO_FINALIZE,
                "avg_conf": avg_conf,
                "tare_status": self.tare_status_text,
                "tare_state": self.tare_state,
                "result_active": self.result_active,
                "last_result": self.last_result,
                "last_result_label": result_label_map.get(self.last_result, self.last_result),
                "stats": self.stats,
                "history": self.history,
                "cycle_status": self.cycle_status_text,
                "ai_status": self.ai_status_text,
                "vl53_1_status": self.vl53_1_status_text,
                "vl53_2_status": self.vl53_2_status_text,
                "feeder_status": self.feeder_status_text,
                "runtime_seconds": int(current_runtime),
                "work_hours": f"{work_hours:.2f}".replace(".", ","),
                "work_hours_val": work_hours,
                "total_fruits": total_fruits,
                "total_weight_kg": total_weight_kg,
                "avg_throughput": throughput,
                "tof1_dist": self.tof1_dist,
                "tof2_dist": self.tof2_dist,
                "feeder_s3_angle": self.feeder_s3_angle,
                "feeder_s4_angle": self.feeder_s4_angle,
                "servo1_angle": self.servo1_angle,
                "servo2_angle": self.servo2_angle,
                "active_preset": self.active_preset,
                "config": {
                    "conf": self.conf_th,
                    "iou": self.iou_th,
                    "imgsz": self.imgsz,
                    "scale": self.scale_factor,
                    "type12": self.type12_th,
                    "type3_min": self.type3_min_th,
                    "speed": self.speed,
                    "servo_ms": self.servo_ms,
                    "preset": self.active_preset,
                },
            }


# Khởi tạo singleton engine
engine = TomatoSorterEngine()


# ==========================================================
# CÁC ROUTE CỦA STARLETTE WEB APP
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
if not os.path.isdir(WEB_DIR):
    WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "web"))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

async def homepage(request):
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

async def video_feed(request):
    def frame_generator():
        while True:
            if engine.running and engine.latest_encoded_frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + engine.latest_encoded_frame + b"\r\n"
                )
                time.sleep(0.02)
            else:
                time.sleep(0.1)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine.active_websockets.add(websocket)

    # Gửi dữ liệu ban đầu
    try:
        await websocket.send_json({
            "type": "PORTS_LIST",
            "ports": engine.get_available_ports(),
            "selected": engine.com_port,
        })
        await websocket.send_json({
            "type": "CAMERAS_LIST",
            "cameras": engine.get_available_cameras(),
            "selected": str(engine.camera_index),
        })
        await websocket.send_json({
            "type": "STATE_UPDATE",
            "state": engine.get_full_state(),
        })

        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "TOGGLE_RUN":
                if engine.running:
                    engine.stop()
                    await websocket.send_json({"type": "TOAST", "message": "Đã dừng hệ thống", "level": "info"})
                else:
                    cam_idx = data.get("camera_index", 0)
                    ok, msg = engine.start(cam_idx)
                    await websocket.send_json({"type": "TOAST", "message": msg, "level": "success" if ok else "error"})

            elif action == "TOGGLE_SERIAL":
                if engine.ser and engine.ser.is_open:
                    engine.disconnect_serial()
                    await websocket.send_json({"type": "TOAST", "message": "Đã ngắt kết nối ESP32", "level": "info"})
                else:
                    port = data.get("port", "")
                    ok, msg = engine.connect_serial(port)
                    await websocket.send_json({"type": "TOAST", "message": msg, "level": "success" if ok else "error"})

            elif action == "REFRESH_PORTS":
                ports = engine.get_available_ports()
                await websocket.send_json({
                    "type": "PORTS_LIST",
                    "ports": ports,
                    "selected": engine.com_port,
                })

            elif action == "REFRESH_CAMERAS":
                cams = engine.get_available_cameras()
                await websocket.send_json({
                    "type": "CAMERAS_LIST",
                    "cameras": cams,
                    "selected": str(engine.camera_index),
                })

            elif action == "TOGGLE_ROI":
                engine.show_roi = bool(data.get("show", False))

            elif action == "SNAPSHOT":
                if engine.latest_encoded_frame is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
                    with open(snap_path, "wb") as f:
                        f.write(engine.latest_encoded_frame)
                    await websocket.send_json({
                        "type": "TOAST",
                        "message": f"Đã lưu ảnh mẫu: snapshot_{timestamp}.jpg",
                        "level": "success",
                    })
                else:
                    await websocket.send_json({
                        "type": "TOAST",
                        "message": "Không có khung hình để chụp",
                        "level": "error",
                    })

            elif action == "RESET_STATS":
                engine.stats = {
                    "type1_weight": 0.0,
                    "type1_count": 0,
                    "type2_weight": 0.0,
                    "type2_count": 0,
                    "type3_weight": 0.0,
                    "type3_count": 0,
                    "reject_count": 0,
                    "reject_weight": 0.0,
                }
                engine.history = []
                engine.accumulated_runtime = 0.0
                engine.session_start_time = time.time() if engine.running else None
                await websocket.send_json({"type": "TOAST", "message": "Đã đặt lại bộ đếm sản lượng và thời gian phiên", "level": "info"})

            elif action == "CLEAR_HISTORY":
                engine.history = []
                await websocket.send_json({"type": "TOAST", "message": "Đã xóa toàn bộ nhật ký phân loại", "level": "info"})

            elif action == "SAVE_CONFIG":
                cfg = data.get("config", {})
                engine.conf_th = float(cfg.get("conf", engine.conf_th))
                engine.iou_th = float(cfg.get("iou", engine.iou_th))
                engine.imgsz = str(cfg.get("imgsz", engine.imgsz))
                engine.type12_th = float(cfg.get("type12", engine.type12_th))
                engine.type3_min_th = float(cfg.get("type3_min", engine.type3_min_th))
                engine.scale_factor = float(cfg.get("scale", engine.scale_factor))
                engine.speed = int(cfg.get("speed", engine.speed))
                engine.servo_ms = int(cfg.get("servo_ms", engine.servo_ms))
                if "preset" in cfg:
                    engine.active_preset = str(cfg["preset"])
                engine.save_config_file()
                engine.send_all_config()
                await websocket.send_json({"type": "TOAST", "message": "Đã lưu vĩnh viễn và gửi cấu hình xuống ESP32", "level": "success"})

            elif action == "RESTORE_DEFAULTS":
                engine.restore_defaults()
                await websocket.send_json({"type": "TOAST", "message": "Đã khôi phục cài đặt mặc định thành công", "level": "info"})

            elif action == "MANUAL_TEST":
                cmd = data.get("test_cmd", "")
                if cmd == "TEST_TARE":
                    engine.send("TARE")
                elif cmd == "TEST_SERVO1_35":
                    engine.send("TEST_SERVO 1 35")
                elif cmd == "TEST_SERVO1_90":
                    engine.send("TEST_SERVO 1 90")
                elif cmd == "TEST_SERVO1_145":
                    engine.send("TEST_SERVO 1 145")
                elif cmd == "TEST_SERVO2_REJECT":
                    engine.send("TEST_SERVO 2 0")
                elif cmd == "TEST_SERVO3_FEED" or cmd == "TEST_FEED":
                    engine.send("FEED")
                elif cmd == "TEST_SERVO3_OPEN":
                    engine.send("TEST_SERVO 3 90")
                elif cmd == "TEST_SERVO3_CLOSE":
                    engine.send("TEST_SERVO 3 0")
                elif cmd == "TEST_SERVO3_HOME":
                    engine.send("TEST_SERVO 3 20")
                elif cmd == "TEST_SERVO4_OPEN":
                    engine.send("TEST_SERVO 4 90")
                elif cmd == "TEST_SERVO4_CLOSE":
                    engine.send("TEST_SERVO 4 0")
                elif cmd == "TEST_SERVO4_HOME":
                    engine.send("TEST_SERVO 4 20")
                elif cmd == "CHECK_SENSORS":
                    engine.send("CHECK")

    except WebSocketDisconnect:
        engine.active_websockets.discard(websocket)
    except Exception:
        engine.active_websockets.discard(websocket)


# Broadcast state đều đặn 40ms (~25 FPS UI updates)
async def state_broadcaster():
    while True:
        if engine.active_websockets:
            full_state = engine.get_full_state()
            msg = json.dumps({"type": "STATE_UPDATE", "state": full_state})
            disconnected = []
            for ws in list(engine.active_websockets):
                try:
                    await ws.send_text(msg)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                engine.active_websockets.discard(ws)
        await asyncio.sleep(0.04)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(state_broadcaster())
    yield
    task.cancel()
    engine.stop()

async def export_csv(request):
    lines = ["\ufeffSTT,Thời gian,Mã hiệu,Phân loại,Khối lượng,Tin cậy,Máng phân loại,Trạng thái"]
    for idx, item in enumerate(engine.history, 1):
        lines.append(
            f'{idx},"{item.get("time","")}","{item.get("badge","")}","{item.get("full_name","")}","{item.get("weight","")}","{item.get("conf","")}","{item.get("servo","")}","{item.get("status_tag","")}"'
        )
    csv_data = "\n".join(lines)
    return HTMLResponse(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=nhat_ky_phan_loai.csv"}
    )

routes = [
    Route("/", homepage),
    Route("/video_feed", video_feed),
    Route("/export_csv", export_csv),
    WebSocketRoute("/ws", websocket_endpoint),
    Mount("/static", StaticFiles(directory=WEB_DIR), name="static"),
]

app = Starlette(
    routes=routes,
    lifespan=lifespan,
)


# ==========================================================
# KHỞI CHẠY ỨNG DỤNG WEB TỰ ĐỘNG
# ==========================================================
def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 60)
    print("  🍅 HỆ THỐNG PHÂN LOẠI CÀ CHUA - GIAO DIỆN WEB DASHBOARD")
    print(f"  🔗 Đang khởi chạy tại: {url}")
    print("=" * 60)

    # Tự động mở trình duyệt sau 1.2 giây
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    # Chạy máy chủ Uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

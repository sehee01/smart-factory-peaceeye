import requests
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class BackendClient:
    """백엔드 서버와 통신하는 클라이언트"""
    
    def __init__(self, backend_url="http://localhost:5000"):
        self.backend_url = backend_url
        self.workers_endpoint = f"{backend_url}/workers"
        self.violations_endpoint = f"{backend_url}/violations"
    
    def _convert_numpy_types(self, obj):
        """NumPy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def send_worker_data(self, worker_data):
        """워커 데이터를 백엔드로 전송"""
        try:
            response = requests.post(
                self.workers_endpoint,
                data=json.dumps(worker_data, cls=NumpyEncoder),  # 커스텀 인코더 사용
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                print(f"[BACKEND] Worker data sent successfully")
                return True
            else:
                print(f"[BACKEND] Failed to send worker data. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[BACKEND] Error sending worker data: {e}")
            return False
        except Exception as e:
            print(f"[BACKEND] JSON serialization error: {e}")
            return False
    
    def send_violation_data(self, violation_data):
        """위반 데이터를 백엔드로 전송"""
        try:
            converted_data = self._convert_numpy_types(violation_data)
            response = requests.post(
                self.violations_endpoint,
                json=converted_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                print(f"[BACKEND] Violation data sent successfully")
                return True
            else:
                print(f"[BACKEND] Failed to send violation data. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[BACKEND] Error sending violation data: {e}")
            return False
        except Exception as e:
            print(f"[BACKEND] JSON serialization error: {e}")
            return False
    
    def send_detections(self, detections):
        """Detection 데이터를 백엔드로 전송"""
        try:
            converted_data = self._convert_numpy_types(detections)
            response = requests.post(
                self.workers_endpoint,
                json=converted_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                print(f"[BACKEND] Detection data sent successfully")
                return True
            else:
                print(f"[BACKEND] Failed to send detection data. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[BACKEND] Error sending detection data: {e}")
            return False
        except Exception as e:
            print(f"[BACKEND] JSON serialization error: {e}")
            return False

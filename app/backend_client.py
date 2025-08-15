import requests
import time


class BackendClient:
    """백엔드 서버와 통신하는 클라이언트"""
    
    def __init__(self, backend_url="http://localhost:5000"):
        self.backend_url = backend_url
        self.workers_endpoint = f"{backend_url}/workers"
        self.violations_endpoint = f"{backend_url}/violations"
    
    def send_worker_data(self, worker_data):
        """워커 데이터를 백엔드로 전송"""
        try:
            response = requests.post(
                self.workers_endpoint,
                json=worker_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"[BACKEND] Worker data sent successfully: {worker_data}")
                return True
            else:
                print(f"[BACKEND] Failed to send worker data. Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[BACKEND] Error sending worker data: {e}")
            return False
    
    def send_violation_data(self, violation_data):
        """위반 데이터를 백엔드로 전송"""
        try:
            response = requests.post(
                self.violations_endpoint,
                json=violation_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"[BACKEND] Violation data sent successfully: {violation_data}")
                return True
            else:
                print(f"[BACKEND] Failed to send violation data. Status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[BACKEND] Error sending violation data: {e}")
            return False


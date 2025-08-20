# smart-factory-peaceseye

> **비전 AI 기반 실시간 안전·작업 모니터링 플랫폼**  
> 기존 **CCTV**와 **Web** 만으로 **PPE 감지, 작업자 추적(ReID), 위험 구역 감시**를 제공하고, **Unity WebGL** 디지털 트윈으로 현장을 시각화합니다. **React**는 로그인/권한 셸을 담당합니다.

Repo: https://github.com/sehee01/smart-factory-peaceeye

---

## 설치해야 할 것들(사전 준비)

### 공통 필수
- **Git**
- **Node.js LTS ≥ 18** (권장 18.x 또는 20.x), **npm**
- **Python 3.10+** (**conda 사용 안 함** → 표준 **venv** 사용)
- **FFmpeg** (AI가 동영상 입력을 읽기 위함)
- **SQLite3** (로컬 점검 시 CLI 편의용 — 서버는 자동으로 DB 파일 생성)
- **Unity Hub + Unity Editor (권장: 6.0)** 및 **WebGL Build Support** 모듈
- **Redis** 서버 — 확장/캐시/멀티 인스턴스 구성 시
- **NVIDIA GPU + 드라이버** — 실시간 추론 성능 향상 (PyTorch 빌드와 호환되는 드라이버/CUDA 런타임)

### OS별 설치 예시
> 아래 예시는 가이드일 뿐, 사내 표준에 맞춰 설치해도 됩니다.

**Ubuntu 22.04+**
```bash
sudo apt update
sudo apt install -y build-essential python3 python3-venv python3-pip \
  ffmpeg sqlite3 pkg-config
# Node LTS
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
```

**macOS (Homebrew)**
```bash
xcode-select --install             # Command Line Tools
brew install node ffmpeg sqlite
```

**Windows 10/11 (CMD)**  
관리자 권한 **CMD**에서 실행:
```cmd
winget install OpenJS.NodeJS.LTS
winget install Git.Git
winget install Python.Python.3.10
winget install Gyan.FFmpeg
winget install SQLite.SQLite
:: (네이티브 모듈 대비 선택) VS 2022 Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools
```
> **주의(Windows)**: 네이티브 모듈을 쓴다면 `node-gyp` 빌드 체인이 필요합니다. 문제가 생기면 `python`, `msbuild`, `cl.exe` 가 PATH에 있는지 확인하세요.  
> `winget`은 **CMD**에서도 동작합니다.

### 버전 확인(권장)
```bash
node -v && npm -v
python --version
ffmpeg -version | head -n 1
sqlite3 --version
# GPU 확인
nvidia-smi   # 리눅스/윈도우 NVIDIA 환경
```

---

## Redis 설치 및 서버 실행

### Redis 설치

#### Windows
1. **Redis 공식 홈페이지에서 다운로드**
   - https://redis.io/download 에서 Windows용 Redis 다운로드
   - 또는 https://github.com/microsoftarchive/redis/releases 에서 최신 Windows 버전 다운로드

2. **설치 후 실행**
   ```cmd
   # Redis 서버 시작 (관리자 권한 CMD에서)
   redis-server
   
   # Redis CLI 실행
   # Redis 설치 폴더의 programs 폴더에서
   cd "C:\Program Files\Redis\programs"
   redis-cli.exe
   ```

#### macOS
```bash
# Homebrew로 설치
brew install redis

# Redis 서버 시작
brew services start redis
# 또는 수동으로
redis-server
```

#### Linux (Ubuntu/Debian)
```bash
# 패키지 매니저로 설치
sudo apt update
sudo apt install redis-server

# Redis 서버 시작
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

---

## 실행 방법(먼저 읽기)

### 실행 순서 요약
1) **백엔드**(Express + WebSocket + SQLite) 실행 → `http://localhost:5000`, `ws://localhost:8000`  
2) **프론트엔드** 실행  
   - React(로그인 셸) 개발 서버: `http://localhost:3000`  
   - Unity WebGL(디지털 트윈): WebGL **빌드 후** 서버 정적 경로에 배포 또는 별도 호스팅
3) **redis서버** 실행   
4) **AI 파이프라인** 실행 (YOLOv11n + ByteTrack + ReID) → **POST**(`/workers`,`/violations`) / **WS**는 Unity가 **수신**

---

### 1) 백엔드 실행 (Express + WebSocket + SQLite)

#### 설치 & 환경설정
```bash
git clone https://github.com/sehee01/smart-factory-peaceeye.git
cd smart-factory-peaceeye/backend   # 실제 경로에 맞게 조정
npm install

# .env 생성
cat > .env << 'EOF'
SERVER_PORT=5000
WS_PORT=8000
CORS_ORIGIN=http://localhost:3000
SQLITE_PATH=./data/peaceeye.db
JWT_SECRET=dev-secret
EOF
```

#### 실행 & 확인
```bash
node app.js
# [HTTP] http://localhost:5000
# [WS  ] ws://localhost:8000

> **포트**: REST=`5000`, WS=`8000` (서로 분리).  
> **DB**: 처음 실행 시 `./data/peaceeye.db` 자동 생성(스키마는 `models/initDB`에서 초기화).

---

### 2) 프론트엔드 실행

#### 2-1. React(로그인/권한 셸)
- 역할: 로그인/로그아웃 & RBAC, Unity WebGL 빌드 파일 **호스팅 셸**, **WS 직접 연결하지 않음**

```bash
cd ../frontend         # 실제 경로에 맞게 조정
npm install
npm run dev            # http://localhost:3000
```

- 로그인 성공 시 **JWT를 postMessage로 Unity에 전달** → Unity가 `ws://localhost:8000?token=...`로 연결

#### 2-2. Unity WebGL(3D/디지털 트윈)
- Unity 에디터에서 프로젝트 열기 → **Build Settings → WebGL → Build**  
- 빌드 산출물(`index.html`, `Build/*`)을 **서버 정적 경로**로 복사하거나 CDN/Nginx에 배포
  - Express 정적 서빙을 쓸 경우, `server/app.js`의 정적 서빙 주석을 해제하고 경로를 Unity 빌드 폴더로 지정
- 토큰 브릿지 예시(React → Unity):
  - **React**: `<iframe src="/unity/index.html">` 로드 후 `postMessage({ jwt })`
  - **Unity index.html**: `window.addEventListener('message', e => window.__JWT__ = e.data.jwt)`

> **브라우저 테스트**: React 셸에서 Unity 페이지로 이동 → 디지털 트윈이 로드되고, WS 수신 시 **워커·알림** UI가 갱신되는지 확인

---

### 3) AI 파이프라인 실행 (**venv 사용, conda 없이**)

#### 3-1. 가상환경 만들기
**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (CMD)**
```cmd
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

#### 3-2. 패키지 설치
```bash
# 프로젝트 requirements.txt 사용 (권장)
pip install -r app/requirements.txt

```

#### 3-3. 실행 
```bash
app/config settings.py에서 모든 설정사항 설정 ( 임계값, 영상주소 등 )
app/reid 에서 python pre_registration.py 실행 -> pre_img 폴더 안에있는 이미지들 redis에 저장
app/ 에서 python new_main.py실행 -> 백엔드로 위치정보 PPE위반 사항 전달
new_main.py , new_main_ultar.py 는 백엔드로 위치전송용 이여서 gui 화면 출력 이루어지지 않음
new_main_ultra_gui.py 파일을 실행하면 영상에서 처리되는 모습을 확인할 수 있음
```
> 파이프라인: **YOLOv11n 검출 → ByteTrack 추적 → OSNet-IBN ReID → Homography 좌표 변환** →  
> `{ class, track_id, ppe_status, real_coordinates, timestamp }` JSON을 **/workers**, **/violations**로 POST.
---

## 포트/엔드포인트 요약
- REST: `http://localhost:5000` → `/`, `/workers`, `/zones`, `/violations`, `/dashboard/*`, `/tasks/*`
- WS:   `ws://localhost:8000` → 서버가 `{ type, payload }`로 브로드캐스트(클라이언트는 수신 전용)

---

## CSV 내보내기
- **WebGL(브라우저)**: `GET /dashboard/export-csv?hours=24` → `Content-Disposition: attachment`
- **Editor/Standalone**: 응답 바이트를 `Application.persistentDataPath`에 저장

---

## 보안/운영 체크리스트
- **WS**: JWT 인증, 하트비트(ping/pong), 백프레셔, 33ms 배치/델타 전송
- **DB**: WAL, 시간 인덱스, UNIQUE 키로 idem 보장, 실패 시 재시도/사후 큐
- **Unity**: 풀링/LOD/압축/보간, HUD에 FPS/p95 WS latency 표시
- **React**: ProtectedRoute, 코드 스플리팅, 캐시/헤더

---

## 참고/감사의 말
- **ByteTrack** — https://github.com/FoundationVision/ByteTrack  
- **deep-person-reid (OSNet-IBN)** — https://github.com/KaiyangZhou/deep-person-reid

---

## 라이선스
(TBD) 저장소 루트의 `LICENSE`를 확인하세요.

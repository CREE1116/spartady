# SPARTADY

교안·강의 필기·음성 전사를 연결하고, 시험까지 남은 시간에 맞춰 일별 학습 계획을 만드는 Electron 앱입니다.

- 검은색 화면과 상단의 **강의 / 캘린더 / 학습** 탭
- PDF 교안과 강의 전체를 기록하는 Markdown 필기장
- Qwen3-ASR 로컬 음성 전사, 녹음 시점의 교안 페이지 연결
- 개념·난이도·근거가 포함된 선수 관계 후보 그래프
- 예산 변경 시 CP-SAT로 다시 계산하는 일별 계획
- 원문 링크를 제공하는 RAG 대화, 3문항 퀴즈와 완료 기록
- 세션별 학습 기록: 앱 재실행 후 대화 이어보기, 새 학습 세션, 퀴즈 결과 보존
- **설정의 지식 정리용 모델**과 **학습창의 대화·퀴즈용 모델**을 독립 선택
- Hugging Face의 지원되는 GGUF 체크포인트 다운로드 → 자체 LMX 변환·무결성 검증

실험 코드, 실험 결과, 실제 교재, 개인 기록, 모델 가중치는 이 저장소에 포함하지 않습니다. LOOM 설치나 서버를 호출하지 않습니다.

## 설치 후 시작하기

1. 설정에서 Hugging Face 모델을 다운로드·변환하고 지식 정리용 모델로 선택합니다. 예시 `unsloth/Qwen3.5-0.8B-GGUF`는 작은 동작 확인용 모델이며, 학습 품질은 더 큰 모델에서 확인하세요.
2. 강의 탭에서 교안을 추가하고 **강의 개념 정리**를 실행합니다.
3. 캘린더에서 시험 날짜·교안 범위·하루 학습시간을 입력합니다.
4. 학습 탭에서 배정된 개념을 선택합니다. 대화창 상단에서 학습 모델을 선택할 수 있습니다.
5. 대화 후 **학습을 마쳤어요**를 누르고 퀴즈 3문항을 모두 맞히면 완료가 기록되고 계획이 갱신됩니다.
6. 음성 전사는 설정의 **전사 환경 설치**를 한 번 실행한 뒤 사용합니다. 첫 모델 로딩에는 가중치 다운로드가 필요합니다.

OpenAI API 키는 설정에서 입력하며 Electron의 OS 암호화 저장소에 보관합니다. OpenAI 로그인 방식은 설치된 Codex CLI의 ChatGPT 로그인을 사용합니다. 앱은 OAuth 토큰 파일을 읽지 않습니다.

## 지원 범위

macOS Apple Silicon: Metal 기반 LMX 추론 및 MLX 기반 Qwen3-ASR. Windows x64: CPU 기반 LMX 추론 및 PyTorch Qwen3-ASR; 호환 CUDA 환경이 있으면 ASR에서 CUDA를 사용합니다. 일반 Hugging Face 모델 전체를 변환하는 도구는 아닙니다. 지원 아키텍처·양자화·텐서 구조 검사를 통과한 Qwen3.5 및 LFM2 계열의 단일 GGUF 파일만 앱 설치 경로에서 받습니다. 일반 Qwen3 MLX 체크포인트는 변환 규격은 지원하지만 현재 추론 엔진에서는 실행되지 않으므로 설치를 거절합니다.

설치 파일에는 Python 학습 엔진, LMX 추론·변환 실행 파일, 전사 환경 설치용 uv를 묶습니다. 기존 소스 폴더나 별도 Python 설치 없이 기본 엔진을 실행합니다. 모델·전사 환경은 사용자 데이터 폴더에 설치됩니다. 현재 배포는 인증서 서명·Apple 공증을 포함하지 않습니다.

## 개발 및 설치 파일 빌드

Python 3.12, Node 22, Rust stable 및 운영체제 C++ 빌드 도구가 필요합니다.

```sh
python -m venv .venv
# macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt pyinstaller uv
python -m unittest discover -s tests
cargo build --release --locked --manifest-path integrations/local_lmx/Cargo.toml
python desktop/scripts/package-runtime.py
cd desktop
npm ci
npm run build
node ../tests/installer-smoke.cjs
npm run package:mac  # Apple Silicon DMG
# Windows에서는 npm run package:win (NSIS EXE)
```

`.github/workflows/release.yml`은 macOS와 Windows 각각에서 엔진·런타임·UI를 검증하고 설치 파일을 만듭니다. 태그 빌드는 GitHub Releases 초안에 파일을 업로드합니다.

## 설계와 한계

[전체 아키텍처 및 RAG](docs/architecture.md)를 참고하세요. 검색은 문자 TF-IDF와 개념 그래프를 결합합니다. 현재 벡터 데이터베이스나 문장 임베딩 검색을 사용하지 않습니다. 선수 관계는 모델이 추출한 후보이며 교육적 정답으로 검증되지 않았습니다. 계획의 제약 충족과 실제 학습 효과는 별개입니다. 수식·도표만으로 표현된 지식은 현재 텍스트 추출에서 누락될 수 있습니다.

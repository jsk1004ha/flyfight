# MaleCNS v1.0 활용 현황

FlyFight는 공식 MaleCNS v1.0의 주석과 뉴런 간 연결 가중치를 로컬 CSR 그래프로 변환한다.

## 현재 포함된 것

- 원본: `data/mcns/v1.0/raw/`
- 변환본: `data/mcns/v1.0/processed/`
- 뉴런: 배포 주석에서 `superclass`가 있는 166,700개 전부
- 방향성 뉴런 간 연결: 25,582,938개
- 시냅스 접촉 합계: 124,177,617
- 공식 body ID, 주석, 방향, 연결 강도를 보존
- `numpy` 메모리맵 CSR이라 전체 그래프를 한꺼번에 RAM에 복사하지 않고 조회 가능

논문과 Google 소개 글은 166,691개를 보고하지만 v1.0 주석 파일에는 `superclass`가 있는 행이 166,700개다. 임의로 9개를 버리지 않고 배포 파일을 보존하며, 이 차이는 `manifest.json`에 기록한다.

## 아직 포함되지 않은 것

- 이 데이터는 구조적 배선도이지 실시간 뉴런 활동 기록이 아니다.
- 흥분성/억제성 부호, 막전위 동역학, 감각 입력 인코딩, 운동 출력 디코딩은 연결 가중치 파일만으로 정해지지 않는다.
- 따라서 현재 PPO 정책의 768 hidden unit가 자동으로 166,700개의 생물학적 뉴런으로 바뀐 것은 아니다.
- 참고 이미지처럼 실제 가지 형태를 그리려면 공식 skeleton 데이터를 별도로 내려받아 단계별 LOD와 바이너리 전송을 구현해야 한다.

## 재생성 및 확인

```powershell
python import_mcns.py data/mcns/v1.0/raw data/mcns/v1.0/processed --overwrite
python -c "from flyfight.connectome import Connectome; c=Connectome.load('data/mcns/v1.0/processed'); print(c.neuron_count, c.edge_count)"
```

기대 출력은 `166700 25582938`이다. 원본 파일 해시와 출처 URL은 `data/mcns/v1.0/processed/manifest.json`에 저장된다.

## 게임에 실제로 연결하는 다음 단계

1. neurotransmitter 표를 추가해 연결 부호에 대한 근거를 보존한다.
2. 시각 감각 뉴런과 descending/motor 뉴런의 명시적 입출력 어댑터를 만든다.
3. 166,700 상태를 모두 갱신하는 희소 동역학을 별도 실험 모드로 추가한다.
4. 기존 synthetic RNN과 같은 시드/상대/맵으로 승률, 명중률, 헤드샷률, 학습 속도를 비교한다.
5. skeleton LOD와 활동량 바이너리 스트림을 추가해 전체 뉴런 시각화를 구현한다.

이 단계를 거치기 전에는 “실제 초파리 뇌가 FPS 전투를 학습했다”고 표현하면 안 된다.

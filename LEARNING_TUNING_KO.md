# 뉴런 상세화와 학습 속도

## 무엇을 빠르게 할 것인가

학습 처리량은 초당 환경 스텝 수(env_steps_s), 전투 학습 효율은 고정 상대에 대한 승률과 무승부율로 확인합니다. 관전 8배속은 별도 관전 경기를 빠르게 재생하는 설정입니다. 학습 처리량을 높이려면 학습 배치와 계산 경로를 조정해야 합니다.

## 현재 구조에서 중요한 병목

- 시야: 매 스텝 두 에이전트의 RGB 광선을 맵의 장애물·벽과 교차 검사합니다. 기본 24×16은 에이전트당 384개 광선입니다. 48×32로 바꾸면 광선이 4배가 되고 입력층 계산도 증가합니다.
- PPO: 기존에는 시퀀스의 매 시점마다 입력층과 출력층을 따로 호출했습니다. 시간축으로 묶을 수 있는 연산은 묶고, 이전 숨은 상태에 의존하는 순환 연산만 시간 순서대로 처리합니다.
- 모델 크기: dense recurrent 가중치는 hidden의 제곱에 비례합니다. 768→1536이면 해당 행렬 크기는 4배, 768→4096이면 약 28.4배입니다. 화면을 풍성하게 만들 목적으로 학습 모델부터 키우는 것은 비용이 큽니다.
- 학습 신호: 승리 +1, 패배 -1, 무승부 0이므로 무승부가 대부분인 구간에서는 유용한 승패 경험이 드뭅니다. 명중률과 결정된 경기 수를 처리량과 함께 봐야 합니다.
- 기억 길이: 기본 horizon=32, dt=0.1이므로 한 번의 역전파가 연결하는 구간은 3.2초입니다. 숨은 상태는 다음 구간으로 이어지지만 기울기는 구간 경계에서 끊깁니다. horizon을 늘리면 긴 행동을 연결해 학습할 여지가 생기지만 메모리와 업데이트 지연이 증가합니다.

## 같은 조건으로 속도 측정

이번 CPU 실측: hidden=768, envs=8, horizon=16, minibatch=8, threads=4, 워밍업 1회 후 3회 중앙값입니다.

| 항목 | 이전 경로 | 묶음 계산 경로 |
|---|---:|---:|
| 환경 스텝/초 | 195.14 | 283.45 |
| PPO 최적화 시간 중앙값 | 0.333초 | 0.140초 |
| 경험 수집 시간 중앙값 | 0.320초 | 0.309초 |

전체 처리량은 약 1.45배였습니다. 짧은 로컬 측정이며 학습 승률 향상을 측정한 결과는 아닙니다. 결과 원본은 `evidence/training_before.json`, `evidence/training_after.json`입니다. hidden=128의 고정 데이터 PPO 비교에서도 7회 중앙값 기준 약 2.96배였으며 이는 최적화 단계만 측정한 값입니다.

묶음 계산은 기본 활성화되어 있습니다. 모델 파라미터 이름과 크기가 유지되어 스키마 3 체크포인트를 로드할 수 있습니다. 부동소수점 계산 순서가 달라 이전 코드와 학습 궤적이 비트 단위로 같다는 보장은 없습니다.

기존 실험 폴더를 변경하지 않고 벤치마크용 모델을 메모리에 생성합니다. 워밍업 후 중앙값을 비교합니다. CPU 실험을 동시에 여러 개 실행하면 비교가 왜곡되므로 순서대로 실행하세요.

```powershell
python benchmark.py --device cpu --envs 8 --hidden 768 --horizon 16 --minibatch-envs 8 --threads 2 --updates 5 --output evidence/bench_threads2.json
python benchmark.py --device cpu --envs 8 --hidden 768 --horizon 16 --minibatch-envs 8 --threads 4 --updates 5 --output evidence/bench_threads4.json
```

같은 명령에 `--no-batch-sequence-encoder`를 붙이면 이전 시점별 계산 경로를 비교할 수 있습니다. 로그의 `collection_seconds`, `optimization_seconds`로 병목을 구분하고 `update_shots`, `update_hits`, `update_headshots`, `update_hit_rate`로 해당 업데이트의 명중 통계를 확인합니다. `games`, `wins`, `draws`는 누적값입니다.

CUDA 사용 가능 여부는 현재 실행할 Python에서 확인합니다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA가 사용 가능한 환경에서만 다음 비교를 실행합니다. 메모리 사용량과 처리량을 보고 envs/minibatch-envs를 선택합니다. BF16 지원 GPU에서 AMP 결과를 일반 정밀도 결과와 비교할 수 있습니다.

```powershell
python benchmark.py --device cuda --envs 64 --minibatch-envs 64 --updates 5 --output evidence/bench_cuda64.json
python benchmark.py --device cuda --envs 128 --minibatch-envs 64 --updates 5 --output evidence/bench_cuda128.json
python benchmark.py --device cuda --envs 128 --minibatch-envs 64 --amp --updates 5 --output evidence/bench_cuda128_amp.json
```

`--compile-policy`는 설치된 PyTorch의 컴파일러 지원과 실제 벤치마크를 확인한 뒤 선택합니다. 컴파일 시간도 포함해 짧은 실행에서 이득이 있는지 판단해야 합니다. `--headless`는 HTTP 서버와 관전 프로세스를 생략합니다.

## 전투를 더 빨리 배우기 위한 다음 실험

1. 동일 seed와 학습 스텝 예산으로 현재 설정을 기준선으로 기록합니다. 발사 수, 명중률, 헤드샷 수, 결정된 경기 수를 확인합니다.
2. 무승부가 많다면 먼저 충분한 경험이 모이는지 확인합니다. 작은 맵 사전학습이나 명중 보상은 과제와 학습 조건을 바꾸므로 별도 실험으로 비교해야 합니다.
3. 명중은 발생하지만 정밀 조준이 어려우면 24×16과 48×32 입력을 비교합니다. 해상도 변경은 입력층 크기를 바꾸므로 새 실험이 필요합니다.
4. 긴 탐색 뒤의 승패 연결이 문제라면 horizon 32/64를 비교합니다. 초당 스텝 수와 같은 환경 스텝에서의 승률을 모두 비교합니다.
5. 학습 전 체크포인트를 고정 상대로 삼아 학습 후 A/B를 각각 평가합니다. 여러 seed와 양쪽 시작 역할을 사용하고 전체 승률·무승부율을 보고합니다. 움직이는 자기대전 상대와의 점수만으로 향상을 판정하지 않습니다.

현재 실행 환경의 CUDA 여부와 새 최적화의 실측 결과는 `evidence/training_sequence_benchmark.json` 및 작업 결과를 확인하세요. CPU 결과에서 GPU 가속 배수를 추정하지 않습니다.

# 서버 상시 학습 배포

`Dockerfile`과 `cloud.py`는 브라우저 없이 지속 학습하며 공개 페이지는 읽기 전용 관전이다.

이미지는 Ubuntu 24.04와 Python 3.12를 사용한다. 의존성은 별도 빌드 단계의
가상환경에 설치하며, 최종 실행 이미지에는 pip 설치 도구를 포함하지 않는다.
PyTorch가 사용자 이름을 조회할 수 있도록 UID 10001 전용 계정을 만들고 캐시는
`/tmp/torchinductor`에 둔다. 운영에서는 루트 파일시스템을 읽기 전용으로 유지한다.
기본 이미지 digest가 고정되어 있어도 보안 업데이트 및 Python 의존성이 달라질 수
있으므로 매 빌드마다 최종 이미지에 HIGH/CRITICAL 검사를 실행해야 한다.

배포 전 컨테이너 확인 (PowerShell):

```powershell
Get-Content tests/container_smoke.py -Raw | docker run --rm -i --network none --read-only --tmpfs /tmp:rw,size=128m --cpus 2 --memory 4g --cap-drop ALL --security-opt no-new-privileges flyfight:test python -
```

이 검사는 학습 1회와 체크포인트 저장·복원을 확인한다. 운영 HTTP/WSS와 영구
볼륨 재시작 검증을 대신하지는 않는다.

필수 설정:

- `PORT=8765`
- `FLYFIGHT_PUBLIC_ORIGIN=https://배포주소` (경로 없음)
- `FLYFIGHT_RUN_DIR=/data/flyfight`
- UID 10001이 쓸 수 있는 영구 볼륨을 위 디렉터리에 마운트
- 단일 replica, 재배포 전략 Recreate (동일 체크포인트에 복수 writer 금지)
- CPU 2코어 / 메모리 4GiB를 초기 용량 검증 기준으로 사용
- 충분한 종료 유예 시간을 설정해 현재 PPO 업데이트와 마지막 저장 완료 보장
- 생존 검사 `/health/live`, 준비·공개 상태 검사 `/health/ready` (학습 프로세스와 관전 모델이 준비될 때만 200)

기본은 CPU, hidden768, 환경8개, 0.01 액션 간격, 헤드샷 보너스0.25다. 최신 체크포인트와 config.json이 있으면 같은 학습 조건으로 재개한다. 컨테이너 임시 디스크만으로는 지속 학습 데이터를 보존할 수 없다.

확인 절차: 학습 스텝 증가 → 브라우저 닫은 뒤 증가 확인 → 체크포인트 생성 → 컨테이너 재시작 → 기존 스텝에서 재개 → 실제 HTTPS/WSS 관전 연결.

Raibit의 영구 저장소·CPU/메모리 설정 지원 변경을 운영에 반영한 뒤 설정한다. 실제 서비스의 PVC Bound, 쓰기 권한과 필요한 메모리 할당이 확인되기 전에는 상시 학습 배포 완료로 간주하지 않는다. 이 문서는 서버 자원 지원 또는 실행 성능을 보장하지 않는다. `/health`의 상세 진단 응답은 기존 Host 제한을 유지하고, Pod IP로 접근하는 위 두 검사 경로는 비밀값 없이 상태 boolean만 반환한다.

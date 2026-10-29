# BenchUp 중간 시각 점검

수정 후 현재 공통 성공 사례를 sample ID 순서로 순회하며 물체별 첫 사례만 선택했다. 선택된 다섯 사례는 `backhoe__render_0009`, `batman__render_0006`, `camera__render_0000`, `dannys_cat__render_0010`, `deer__render_0012`다. 각 행은 편집 target 입력과 모델별 raw target-view 렌더를 보여 준다. 이미지와 선택 manifest는 file4 `benchup/reports/common-visual-audit-v2.jpg`, `common-visual-audit-v2.json`에 있다.

앞뒤 축 보정 후 Batman 얼굴과 고양이 얼굴이 TRELLIS.2 계열의 렌더에서 보이는 등 이전의 반전 패턴은 이 사례들에서 개선됐다. 그러나 예를 들어 deer에서는 일부 모델 출력이 입력과 다른 방향으로 보인다. 이 점검은 전 모델·전 사례의 자세를 검증하지 않으며, 그 차이를 특정 모델 오류나 카메라 오류로 단정할 수 없다.

이 다섯 사례에서는 MV-Pixal3D 계열과 Selected가 일부 target 편집을 충분히 재현하지 못하는 모습도 보인다. 예를 들어 카메라의 위쪽 부품이나 고양이의 둥근 형상이 일부 결과에서 약해진다. 반면 단일 target 입력 재구성은 입력 이미지의 편집을 잘 따라가는 예가 있다. 이는 소수 사례의 육안 관찰이며 source 보존과 전체 분포의 우열, 최종 수치 또는 일반화 성능을 결론내릴 근거가 아니다. 최종 평가는 고정 모집단의 공통 성공 집합과 모델별 coverage로 수행한다.

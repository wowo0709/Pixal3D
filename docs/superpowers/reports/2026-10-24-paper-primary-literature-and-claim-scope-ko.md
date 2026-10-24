# 원래 논문의24개 문헌 대조와 현재 claim 범위

2026-10-24. 원래5쪽 PDF의 참고문헌24개를 원저자 제출 arXiv·CVF·OpenReview와 대조했다. 기존에 확인한 핵심6개 문헌의 근거는 재사용하고, 이번에는 나머지19개 원문 identity와 abstract 수준의 task/method 범위를 확인했다. 기존6개 중 VGGT-Omega는 원래24개 목록 밖에 있다. 이는 모든 논문의 세부 구현·실험을 재현하거나 원래 PDF의 모든 문장을 검증했다는 뜻은 아니다.

이번 작업으로 현재 초안의 bibliography를6개에서17개로 확장했다. 직접 관련된11개를 추가했고, 나머지는 아래 대조표에 남겼다. 인용 수를 늘리기 위해 모든 문헌을 본문에 나열하지 않는다. 모든 링크는 저자 제출 원문/학회 원문이다.

## 중요한 정정과 논문 포지셔닝

1. 원래[6]의 UniRecGen, arXiv2604.01479는 현재v3에서 **RecGen3D: Reconstruction-Guided 3D Generation in a Shared Canonical Space**로 제목이 바뀌었다. 현재 제목으로 인용하고 원래 이름과의 대응을 보존한다. [저자 제출 원문](https://arxiv.org/abs/2604.01479).
2. Matching/correspondence-aware attention 자체는 선행 기법이다. MVDiffusion은 주어진 pixel correspondence로 view 간 정보를 결합하고, CorrAdapter는 diffusion의 중간 특징에서 대응을 찾아 matching 영역의 정보를 결합한다. 따라서 이를 우리만의 최초 기여로 쓰지 않는다. [MVDiffusion](https://arxiv.org/abs/2307.01097), [CorrAdapter](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html).
3. TextureDreamer의 target shape, GeoStyle/APAP의 source mesh, SpaceControl의 explicit spatial control은 현재 **edited target RGB + undeformed source RGB3장 → predicted geometry/appearance**와 입력/목적이 다르다. GT target mesh를 추론에 입력한다고 서술하지 않는다. 이 차이만으로 어느 방법보다 우수하다고 결론내리지 않는다. [TextureDreamer](https://arxiv.org/abs/2401.09416), [GeoStyle](https://arxiv.org/abs/2604.07795), [APAP](https://arxiv.org/abs/2311.16739), [SpaceControl](https://arxiv.org/abs/2512.05343).
4. Multi-view projection·reconstruction/generation 결합도 기존 연구에 있다. 현재 기여는 의도적으로 다른 geometric state의 evidence를 pretrained3D conditioning에 전달하는 설계와 그 검증에 한정한다. Semantic 탐색·후보 결합·ray 공유·R의 효과는 실제 matched 비교와 downstream 지표로 확인해야 한다. [ReconViaGen](https://arxiv.org/abs/2510.23306), [RecGen3D](https://arxiv.org/abs/2604.01479), [GenRecon](https://arxiv.org/abs/2605.23888).

## 원래24개 참고문헌 대조

아래 범위는 bibliographic identity 및 해당 task의 요약이다. `본문`은 현재 초안에 실제 인용한 문헌이고, `문맥`은 일반 배경으로 확인했지만 현재 본문의 중심 비교에 추가하지 않은 문헌이다. 학회 page/venue를 새로 확인하지 않은 경우 추정해 BibTeX에 넣지 않고 검증한 arXiv identity를 사용한다.

| 원래 번호 | 확인한 identity/primary source | 역할 | 현재 사용 |
|---|---|---|---|
| 1 | [ReconViaGen,2510.23306](https://arxiv.org/abs/2510.23306) | Reconstruction/generation 결합 | 본문 |
| 2 | [GeoStyle,2604.07795](https://arxiv.org/abs/2604.07795) | Image-guided source-mesh stylization | 본문 |
| 3 | [SpaceControl,2512.05343](https://arxiv.org/abs/2512.05343) | Test-time explicit spatial control | 본문 |
| 4 | [ShapeUP,2602.05676](https://arxiv.org/abs/2602.05676) | Source-shape/image-conditioned editing | 본문·기존 근거 |
| 5 | [MV-Adapter,2412.03632](https://arxiv.org/abs/2412.03632) | Multi-view image generation adapter | 본문 |
| 6 | [현재 RecGen3D,2604.01479v3](https://arxiv.org/abs/2604.01479) | Shared canonical reconstruction/generation; 원래 UniRecGen | 본문·제목 갱신 |
| 7 | [Hunyuan3D2.1,2506.15442](https://arxiv.org/abs/2506.15442) | Shape/texture generative pipeline와 학습 workflow | 문맥 |
| 8 | [Hunyuan3D2.5,2506.16504](https://arxiv.org/abs/2506.16504) | Shape/texture3D generation | 문맥 |
| 9 | [Pixal3D,2605.10922](https://arxiv.org/abs/2605.10922) | Pixel-aligned generation 기반 | 본문·기존 근거 |
| 10 | [TripoSG,2502.06608](https://arxiv.org/abs/2502.06608) | Image-conditioned shape generation | 문맥 |
| 11 | [RoMa-Omega,2609.09507](https://arxiv.org/abs/2609.09507) | 3D representation 기반 matching | 본문·기존 근거 |
| 12 | [GenRecon,2605.23888](https://arxiv.org/abs/2605.23888) | Projected multi-view scene reconstruction | 본문 |
| 13 | [Free-Range Gaussians,2604.04874](https://arxiv.org/abs/2604.04874) | Non-grid-aligned generative Gaussian reconstruction | 문맥 |
| 14 | [MVDream,2308.16512](https://arxiv.org/abs/2308.16512) | Text-conditioned multi-view diffusion | 문맥 |
| 15 | [DIFT,2306.03881](https://arxiv.org/abs/2306.03881) | Diffusion feature correspondence | 본문 |
| 16 | [MVDiffusion,2307.01097](https://arxiv.org/abs/2307.01097) | Given correspondence-aware multi-view diffusion | 본문 |
| 17 | [TRELLIS,2412.01506](https://arxiv.org/abs/2412.01506) | Structured latent3D generation | 문맥 |
| 18 | [TRELLIS.2,2512.14692](https://arxiv.org/abs/2512.14692) | Native geometry/material latent 기반 | 본문·기존 근거 |
| 19 | [RnG,2603.01194](https://arxiv.org/abs/2603.01194) | Reconstruction/generation과 implicit3D completion | 문맥 |
| 20 | [TextureDreamer,2401.09416](https://arxiv.org/abs/2401.09416) | Reference image에서 target-shape texturing | 본문 |
| 21 | [APAP,2311.16739](https://arxiv.org/abs/2311.16739) | Diffusion-prior 기반 mesh deformation | 본문 |
| 22 | [A Tale of Two Features,2305.15347](https://arxiv.org/abs/2305.15347) | Diffusion/DINO correspondence 상보성 | 본문·기존 근거 |
| 23 | [CorrAdapter,CVF2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html) | Native diffusion correspondence와 matching-area aggregation | 본문 |
| 24 | [Hunyuan3D2.0,2501.12202](https://arxiv.org/abs/2501.12202) | Image-conditioned shape/texture generation | 문맥 |

원래[13]의 첫 저자는 현재 arXiv에 Ahan Shabanov로 표시된다. 인용이 필요하면 원래 PDF의 Akhmedkhan 표기를 무조건 복사하지 않고 선택한 primary version의 저자 metadata를 사용한다. 본문에 추가하지 않은11개 문헌의 전체 저자/venue/page를 확인한 것으로 과장하지 않는다.

## 현재 본문에서 쓰는 구분

- Reconstruction/3D generation: TRELLIS.2·Pixal3D·ReconViaGen·RecGen3D·GenRecon.
- Image/mesh-conditioned editing 및 texturing: ShapeUP·GeoStyle·APAP·SpaceControl·TextureDreamer.
- Multi-view/correspondence-aware generation: MV-Adapter·MVDiffusion·CorrAdapter.
- Matching representation: DIFT·diffusion/DINO·VGGT-Omega·RoMa-Omega.

현재 초안에는 이 구분을 반영했다. External method들의 성능 수치를 옮겨와 동일 protocol 비교처럼 나열하지 않았다. 현재 실행하는 zero-shot/V1/R/선택 Ours의 결과만으로 external editing method 전체에 대한 SOTA나 우위를 주장하지 않는다. 원래 PDF의 abstract와 빈 main-table baseline 행은 실제 정량 비교 완료의 근거가 아니다.

이 문헌 대조는 현재 학습·DEV128·기전 관측을 중단하거나 새로운 external baseline 학습을 등록한 작업이 아니다. 최종 구조/metric 선택, matched ablation, seed 확인과 TEST508은 기존 핵심 계획대로 계속한다.

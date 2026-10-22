# R-mediated PBR의 실제 generated support conditioning

새 native PBR source 모듈을 full cascade에 적용하려면 학습 cache의 GT-support
`patch_ids`를 재사용할 수 없다. 실제 Shape1024 prediction의 token 수와 좌표 순서가
GT와 다르므로, 실제 prediction 좌표와 기존 calibrated camera로 주소를 다시 계산한다.

`generated_r_mediated_pbr.py`는 기존 `projection_maps.anchor_patch_indices`의
target-address 규칙을 재사용한다. Source-ray pooling의 native nearest32 규칙과
target cache의 floor/clip 규칙은 각각 원래 역할대로 유지한다. Native bilinear feature
delivery를 nearest overwrite로 바꾸거나 GT visibility write gate를 추가하지 않는다.

새 loader에 넘기는 row는 RGB·camera·scale·image descriptor 경로의12개 field만
포함한다. Actual RGB 파일의 fingerprint를 다시 계산하고 두 cache의 schema/input hash와
Omega23597(E7)을 확인한다. SS cache에서는 FRAME/GLOBAL keys와 source-valid만,
PBR cache에서는 실제 DINO+NAF2048 values와 target query만 읽는다.
Legacy semantic indices/score, GT projection, GT latent/mesh 및 deformation label은 읽지 않는다.
Source-camera member posterior는 실제 native sparse consumer가 online으로 계산한다.

`encode_generated_r_pbr`는 실제 `RMediatedPBRNativeTrainer._encode_deform_views`를
재사용한다. Source 모듈까지 training과 같은 BF16 context에 넣고, native sparse
decoder 밖으로 autocast를 확장하지 않는다. Pipeline hook은 texture extractor에만
적용하며 Shape 단계는 원래 함수로 보낸다. 예외 시 hook과 low-VRAM offload를 복구한다.
CPU/CUDA sampler RNG가 바뀌면 실패하고, NAF는 paired seed reset 전에 미리 로드해야 한다.

## 실제 CPU 입력 검사

- 실제 TRAIN31900 및 DEV GT3401 좌표에서 새 loader의 여섯 image tensor와
  모든 target patch ID가 현재 학습 dataset/cache와 정확히 일치했다.
- 기존 원본 Shape 모델로 생성한 DEV의 실제3085-token support가 GT3401과 다름을 확인하고
  3085개 새 주소를 계산했다. 실제 helper 호출의 `np.load`를 제한해 두 image cache만
  열고 forbidden GT/legacy field를 읽으면 실패하도록 검사했다.
- 기존 raw 데이터는 수정하거나 업로드하지 않았다. 해당 원본 cache/이미지/QA latent는
  file4에서 실제 SHA/size를 읽어 로컬과 비교했다. Canonical QA artifact10개45,417,476byte도
  별도로 file4와 대조했다. GT는 **CPU training-cache parity 검사에만** 사용했다.

근거는 `r-mediated-source-pilots-v1/PBR-generated-R-source-support-v1/CPU-input-QA-v2/`의
`audit.json`, `actual-external-file4-inputs.json`, minimal image row 및 support snapshot이다.
v1은 export manifest가 다른 canonical root의 원본 데이터까지 포함하려 해 실패했고,
실패 로그/출력을 남겼다. v2는 원본을 readonly SSH SHA로 검증하고, 새 QA 결과만
result root에 보관했다. 이 export 문제는 학습이나 데이터 정합 검사의 실패가 아니다.

## GPU proof 범위

`probe_generated_r_pbr_support.py`는 한 GPU에서 native DINO/NAF와 source 세 mode의
conditioning을 비교한다. 실제 TRAIN/DEV training-support parity와 DEV generated support를
구분한다. Cold source의 global/projection/negative가 원래 native pipeline과 같아야 한다.
임시 R output bias=1 개입으로 DINO와 NAF 전달이 모두 실제 consumer cast 이후에도
남고 training consumer와 같은지 검사한다. 이 bias는 trained weight가 아니며, 검사 후
정확히 복원하고 module을 폐기한다. Denoiser/sampler/optimizer/quality metric은 실행하지 않는다.

CPU 입력 일치나 이 GPU conditioning proof는 실제 PBR texture 생성·렌더링 성능의 대체가
아니다. Actual trained source checkpoint를 strict load하는 별도 native replay/cascade와
최종 source/target metric 비교가 여전히 필요하다.

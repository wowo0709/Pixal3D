# Models

## Scope

- Model definitions live here; sparse tensor and backend contracts are in `../modules/sparse/AGENTS.md`.
- Package lazy exports and `from_pretrained` checkpoint metadata belong to the parent `pixal3d/AGENTS.md`.

## Where to look

| Task | Location | Notes |
|---|---|---|
| Dense sparse-structure VAE | `sparse_structure_vae.py` | 3D convolutional encoder/decoder; tensors are dense PyTorch volumes. |
| Sparse-structure flow | `sparse_structure_flow.py` | Backbone consumes dense feature volumes and timestep/condition tensors. |
| Structured-latent flow | `structured_latent_flow.py` | `SLatFlowModel` and elastic variant operate on sparse latent tokens. |
| Sparse U-Net VAE | `sc_vaes/sparse_unet_vae.py` | Sparse encoder/decoder modules. |
| FlexiDualGrid VAE | `sc_vaes/fdg_vae.py` | Dual-grid representation encoder/decoder. |

## Contracts

- `SparseStructureEncoder`/`SparseStructureDecoder` use dense `torch.Tensor` volumes with channel-first 3D convolutions (`Conv3d`). Their latent/output channel counts are constructor parameters.
- `SparseStructureFlowModel.forward(x, t, cond)` is defined for dense tensors; do not substitute `SparseTensor` merely because its name contains “SparseStructure”.
- `SLatFlowModel` builds blocks from `ModulatedSparseTransformerCrossBlock`, projects with `SparseLinear`, and uses sparse latent tokens. Keep token features and coordinate/layout alignment intact through model changes.
- Model classes expose device from their parameters where implemented; dtype conversion for transformer torso goes through `convert_to()` in structured-latent flow.
- VAE representation boundaries are family-specific. Check the matching trainer and `pixal3d/representations` implementation before changing encoder/decoder tensor shape or interpretation.

## Gotchas

- Dense sparse-structure VAE and sparse U-Net VAE are separate implementations with distinct tensor representations.
- Image-projection conditioning is assembled in flow-model blocks; coordinate projection details are shared with trainer conditioning mixins. Inspect both call sites before changing conditioning dimensions.
- Keep model constructor/config keys aligned with the experiment configuration that instantiates these classes.

## MV Initialization

- Calibrated multi-view feature alignment/averaging adapts the condition input;
  it preserves the four official denoiser architectures and checkpoint shapes.
  Treat conditioning changes separately from model/decoder representation changes.

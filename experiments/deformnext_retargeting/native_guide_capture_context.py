"""Record the guides returned by the original Shape decoder without replacing them."""
from contextlib import contextmanager

from .archived_native_guides import save_guides


@contextmanager
def capture_native_guides(pipeline_type, archive):
    previous = pipeline_type.decode_shape_slat
    events = []

    def decode(self, shape, resolution):
        if resolution != 1024 or events:
            raise ValueError('one original native1024 Shape decoder call required')
        meshes, guides = previous(self, shape, resolution)
        identity = save_guides(archive, shape, guides)
        events.append(dict(actual_guide_count=len(guides), tensor_identity=identity))
        return meshes, guides

    pipeline_type.decode_shape_slat = decode
    try:
        yield events
    finally:
        pipeline_type.decode_shape_slat = previous

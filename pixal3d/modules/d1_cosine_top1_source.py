"""Same Top1+R carrier with D1 contextual cosine and no learned scalar head."""
import torch

from .raw_correspondence_top1_source import RawCorrespondenceTop1SourceConditioner


class D1CosineTop1SourceConditioner(RawCorrespondenceTop1SourceConditioner):
    def __init__(self,*,correspondence='semantic',**kwargs):
        if correspondence!='semantic':raise ValueError('D1 cosine comparison uses the same RAW64 proposals')
        super().__init__(correspondence=correspondence,**kwargs)
        self.register_buffer('D1_cosine_top1_source_schema',torch.tensor([1],dtype=torch.uint8))

    def _configure_matcher(self):
        if self.q2c is None:raise ValueError('original frozen D1 checkpoint required')
        self.q2c.fusion_head=None
        self.q2c.requires_grad_(False).eval()
        self.matching_mode='D1-cosine'

    def _semantic_posterior(self,**kwargs):
        from .d1_cosine_member_posterior import d1_cosine_member_ray_posteriors
        return d1_cosine_member_ray_posteriors(self.q2c,**kwargs)

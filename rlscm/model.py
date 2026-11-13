"""CNN feature extractor; SAC actor and critics are provided by SB3."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Encoder(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=69)
        self.cnn = nn.Sequential(nn.Conv2d(3, 8, 7, stride=4), nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())

    def forward(self, obs):
        return torch.cat([self.cnn(obs['image']), obs['state']], dim=1)



class SpatialEncoder(BaseFeaturesExtractor):
    """Larger encoder (~0.7 M params) that keeps spatial detail for thin 1-pixel ridges.

    Max-pooling (not large strides) preserves sparse binary ridges; the output keeps an
    8 x 8 spatial grid (flattened, so ridge positions stay addressable) before a linear
    projection. Experiment E1: is Poisson's ratio poorly recovered because the small
    Encoder (2 x 2 x 16 output) cannot see fine ridge spacing/curvature?
    """

    def __init__(self, observation_space, features_dim=256):
        state_dim = observation_space['state'].shape[0]
        super().__init__(observation_space, features_dim=features_dim + state_dim)

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(), nn.MaxPool2d(2),  # 256 -> 64
            *block(16, 32),    # 32
            *block(32, 64),    # 16
            *block(64, 128),   # 8
            nn.Conv2d(128, 32, 1), nn.ReLU(), nn.Flatten(),                       # 8*8*32 = 2048
            nn.Linear(2048, features_dim), nn.ReLU())

    def forward(self, obs):
        return torch.cat([self.cnn(obs['image'].float()), obs['state']], dim=1)


class EfficientNetV2SEncoder(BaseFeaturesExtractor):
    """WaveDL/torchvision V2-S backbone; scratch three-channel SAC adaptation.

    GroupNorm replaces BatchNorm for small replay batches. No pretrained weights.
    """
    def __init__(self, observation_space, features_dim=256):
        from torchvision.models import efficientnet_v2_s
        super().__init__(observation_space, features_dim=features_dim+observation_space['state'].shape[0])
        self.backbone=efficientnet_v2_s(weights=None, stochastic_depth_prob=0.0)
        def replace_norm(module):
            for name,layer in list(module.named_children()):
                if isinstance(layer,nn.BatchNorm2d):
                    groups=min(32,layer.num_features)
                    while layer.num_features % groups: groups-=1
                    setattr(module,name,nn.GroupNorm(groups,layer.num_features,eps=layer.eps))
                else: replace_norm(layer)
        replace_norm(self.backbone)
        width=self.backbone.classifier[1].in_features
        self.backbone.classifier=nn.Identity()
        self.projection=nn.Sequential(nn.Linear(width,features_dim),nn.ReLU())

    def forward(self, obs):
        return torch.cat([self.projection(self.backbone(obs['image'].float())),obs['state'].float()],dim=1)



class WaveDLEpoch50Encoder(BaseFeaturesExtractor):
    """Frozen WaveDL V2-L epoch50 features; independent grayscale encoding per channel.

    No regression head or label scaler is used. Physical initialization remains random.
    """
    def __init__(self, observation_space, features_dim=256):
        from torchvision.models import efficientnet_v2_l
        from pathlib import Path
        super().__init__(observation_space,features_dim+observation_space['state'].shape[0])
        import os
        default=Path(__file__).resolve().parents[1]/'experiments/models/efficientnet_v2_l_transfer/source'/getattr(self,'source_weights','epoch50.bin')
        path=Path(os.environ.get('WAVEDL_ENCODER_CHECKPOINT',str(default)))
        self.backbone=efficientnet_v2_l(weights=None)
        old=self.backbone.features[0][0]
        self.backbone.features[0][0]=nn.Conv2d(1,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        state=torch.load(path,map_location='cpu',weights_only=True)
        feature_state={}
        for key,value in state.items():
            pos=key.find('features.')
            if pos>=0:feature_state[key[pos:]]=value
        expected={k:v for k,v in self.backbone.state_dict().items() if k.startswith('features.')}
        if set(feature_state)!=set(expected):
            raise ValueError('WaveDL checkpoint backbone keys do not match torchvision V2-L')
        self.backbone.classifier=nn.Identity()
        self.backbone.load_state_dict(feature_state,strict=True)
        self.backbone.requires_grad_(False);self.backbone.eval()
        self.projection=nn.Sequential(nn.Linear(1280*3,features_dim),nn.ReLU())

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'backbone'):self.backbone.eval()
        return self

    def forward(self,obs):
        x=obs['image'].float()
        with torch.no_grad():
            # One batched backbone call; preserve per-sample channel ordering.
            batch=x.shape[0]
            encoded=self.backbone(x.reshape(batch*3,1,*x.shape[-2:]))
            features=encoded.reshape(batch,3*1280)
        return torch.cat([self.projection(features),obs['state'].float()],dim=1)


class WaveDLEpoch100Encoder(WaveDLEpoch50Encoder):
    source_weights = 'epoch100.bin'

ENCODERS = {'small': Encoder, 'spatial': SpatialEncoder, 'efficientnet_v2_s': EfficientNetV2SEncoder, 'wavedl_epoch50': WaveDLEpoch50Encoder, 'wavedl_epoch100': WaveDLEpoch100Encoder}

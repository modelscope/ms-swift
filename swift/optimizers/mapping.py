from .galore import create_galore_optimizer
from .lorap import create_lorap_optimizer
from .multimodal import create_multimodal_optimizer
from .muon import create_muon_optimizer
from .muonclip import create_muon_clip_optimizer

# Add your own optimizers here, use --optimizer xxx to train
optimizers_map = {
    'galore': create_galore_optimizer,
    'lorap': create_lorap_optimizer,
    'muon': create_muon_optimizer,
    'muonclip': create_muon_clip_optimizer,
    'multimodal': create_multimodal_optimizer,
}

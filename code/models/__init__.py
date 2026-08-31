"""Model registry: name -> class. `training.py` / `eval.py` / `inference.py` dispatch by name."""

from models.le_world_model import LeWorldModel
from models.ts_jepa import TimeSeriesJEPA
from models.ebt import EnergyBasedTransformer
from models.simple_transformer import SimpleTransformer
from models.ttt_transformer import TTTTransformer
from models.node_world_model import NeuralODEWorldModel
from models.meta_le_world_model import MetaLeWorldModel
from models.genie_world_model import GenieWorldModel
from models.ts_diffusion import TimeSeriesDiffusion
from models.gnn_world_model import GNNTimeSeries
from models.multihorizon_le_world_model import MultiHorizonLeWorldModel
from models.ttt_val_model import TTTValModel
from models.multihorizon_meta_le_world_model import MultiHorizonMetaLeWorldModel
from models.kan_mh_meta import KANMultiHorizonMeta
from models.neural_sde import (NeuralSDE, SDEKANMultiHorizon, SDEKANMultiHorizonMeta,
                               SDELeWorldLatent)
from models.ebt_le_mh_meta import EBTLeWorldMHMeta
from models.composed import GNNLeMeta, KANGNNMeta, GNNSDE, EBTJepaSDE
from models.neural_pde import (KANNeuralPDE, LeWorldPDE, LeWorldPDEMeta, GNNLeWorldMetaPDE,
                               GNNLeWorldMetaODE)
from models.ada_jepa import AdaJEPA, FNOJEPA, GNNAdaJEPA, GNNFNOJEPA, Mamba2AdaJEPA
from models.rate_family import RateAnchor, HazardMono, TimeWarp, UDEHybrid
from models.dist_family import QuantileHead, TPPEvents, NPEHead, CFPaired
from models.anticollapse import (RateAnchorAC, HazardMonoAC, TimeWarpAC, UDEHybridAC,
                                 QuantileHeadAC, TPPEventsAC, NPEHeadAC, CFPairedAC)

REGISTRY = {
    m.name: m for m in [
        LeWorldModel,
        TimeSeriesJEPA,
        EnergyBasedTransformer,
        SimpleTransformer,
        TTTTransformer,
        NeuralODEWorldModel,
        MetaLeWorldModel,
        GenieWorldModel,
        TimeSeriesDiffusion,
        GNNTimeSeries,
        MultiHorizonLeWorldModel,
        TTTValModel,
        MultiHorizonMetaLeWorldModel,
        KANMultiHorizonMeta,
        NeuralSDE,
        SDEKANMultiHorizon,
        SDEKANMultiHorizonMeta,
        SDELeWorldLatent,
        EBTLeWorldMHMeta,
        GNNLeMeta,
        KANGNNMeta,
        GNNSDE,
        EBTJepaSDE,
        KANNeuralPDE,
        LeWorldPDE,
        LeWorldPDEMeta,
        GNNLeWorldMetaPDE,
        GNNLeWorldMetaODE,
        AdaJEPA,
        FNOJEPA,
        GNNAdaJEPA,
        GNNFNOJEPA,
        Mamba2AdaJEPA,
        RateAnchor,
        HazardMono,
        TimeWarp,
        UDEHybrid,
        QuantileHead,
        TPPEvents,
        NPEHead,
        CFPaired,
        RateAnchorAC,
        HazardMonoAC,
        TimeWarpAC,
        UDEHybridAC,
        QuantileHeadAC,
        TPPEventsAC,
        NPEHeadAC,
        CFPairedAC,
    ]
}


def build(name: str, T: int, scale: float = 1.0):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. known: {sorted(REGISTRY)}")
    return REGISTRY[name](T, scale=scale)

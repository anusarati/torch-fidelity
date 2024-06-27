import torch
import torch.nn as nn
import torch.nn.functional as F
from malgen.infra.classifier import Classifier

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import text_to_dtype, vassert


class FeatureExtractorCustom(FeatureExtractorBase):
    def __init__(
        self,
        name,
        features_list,
        feature_extractor_weights_path=None,
        feature_extractor_internal_dtype=None,
        **kwargs,
    ):
        """ """
        super(FeatureExtractorCustom, self).__init__(name, features_list)
        vassert(
            feature_extractor_internal_dtype in ("float32", "float64", None),
            "Only 32 and 64 bit floats are supported for internal dtype of this feature extractor",
        )
        self.feature_extractor_internal_dtype = text_to_dtype(feature_extractor_internal_dtype, "float32")

        self.classifier = Classifier.load_from_checkpoint(feature_extractor_weights_path)

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        return (self.classifier.classifier(x).pooler_output,)

    @staticmethod
    def get_provided_features_list():
        return "custom"

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": "custom",
            "fid": "custom",
            "kid": "custom",
            "prc": "custom",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return False

    @staticmethod
    def get_dummy_input_for_compile():
        raise NotImplementedError

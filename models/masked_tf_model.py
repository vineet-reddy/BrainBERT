"""
This module implements a Masked Transformer Model for brain signal processing.
The model follows a BERT-style architecture where it learns representations
of brain signals through masked prediction tasks.
"""

from models import register_model
import torch.nn as nn
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput
from models.spec_prediction_head import SpecPredictionHead 

@register_model("masked_tf_model")
class MaskedTFModel(BaseModel):
    """
    A transformer-based model for masked brain signal prediction.
    
    This model processes brain signal spectrograms using a transformer architecture,
    similar to BERT's masked language modeling approach but adapted for continuous
    brain signal data. It consists of:
    1. An input encoding layer that processes raw spectrograms
    2. A transformer encoder for learning contextual representations
    3. A prediction head for reconstructing masked portions of the input

    Configuration Parameters (from conf/model/masked_tf_model.yaml):
        - hidden_dim: 768 - Dimension of transformer hidden states
        - layer_dim_feedforward: 3072 - Dimension of feedforward network in transformer
        - layer_activation: gelu - Activation function used in transformer layers
        - nhead: 12 - Number of attention heads in transformer
        - encoder_num_layers: 3 - Number of transformer encoder layers
        - input_dim: 40 - Dimension of input features
    """
    
    def __init__(self):
        """Initialize the masked transformer model."""
        super(MaskedTFModel, self).__init__()

    def forward(self, input_specs, src_key_mask, intermediate_rep=False, rep_from_layer=-1):
        """
        Forward pass of the model.
        
        Args:
            input_specs: Input spectrograms [batch, seq, input_dim=40]
            src_key_mask: Masking tensor for padding positions
            intermediate_rep: If True, returns intermediate representations
            rep_from_layer: Which transformer layer to extract representations from
                          (-1 means final layer)
        
        Returns:
            output_specs: Predicted spectrogram values [batch, seq, hidden_dim=768]
            pos_enc: Positional encodings used in the model
        """
        input_specs, pos_enc = self.input_encoding(input_specs)
        input_specs = input_specs.transpose(0,1) #nn.Transformer wants [seq, batch, dim]
        if rep_from_layer==-1:
            output_specs = self.transformer(input_specs, src_key_padding_mask=src_key_mask)
        else:
            raise NotImplementedError
        output_specs = output_specs.transpose(0,1) #[batch, seq, dim]
        if intermediate_rep:
            return output_specs
        output_specs = self.spec_prediction_head(output_specs)
        return output_specs, pos_enc

    def init_weights(self, module):
        """
        Initialize the weights of the model.
        
        Applies custom initialization for Linear and LayerNorm layers:
        - Linear layers: Zero out biases if present
        - LayerNorm layers: Zero out biases and set scale to 1.0
        
        Args:
            module: PyTorch module whose weights need to be initialized
        """
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.bias.data.fill_(1.0)

    def build_model(self, cfg):
        """
        Build the complete model architecture based on config.
        
        Args:
            cfg: Configuration object containing model hyperparameters:
                - hidden_dim: 768 - Dimension of transformer hidden states
                - nhead: 12 - Number of attention heads
                - layer_dim_feedforward: 3072 - Dimension of feedforward network
                - layer_activation: gelu - Activation function type
                - encoder_num_layers: 3 - Number of transformer layers
                - input_dim: 40 - Input feature dimension
        """
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim  # Set to 768 in config
        # Input encoding layer processes raw spectrograms (input_dim=40 -> hidden_dim=768)
        self.input_encoding = TransformerEncoderInput(cfg)
        # Build transformer encoder layer with specified parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 768
            nhead=self.cfg.nhead,  # 12 attention heads
            dim_feedforward=self.cfg.layer_dim_feedforward,  # 3072
            activation=self.cfg.layer_activation  # gelu
        )
        # Stack multiple encoder layers (3 layers as specified in config)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.cfg.encoder_num_layers  # 3
        )
        # Final prediction head for spectrogram reconstruction
        self.spec_prediction_head = SpecPredictionHead(cfg)
        # Initialize all model weights
        self.apply(self.init_weights)

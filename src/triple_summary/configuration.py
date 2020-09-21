import logging

logger = logging.getLogger(__name__)


class DecoderConfig(object):
    def __init__(
        self,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        dec_dropout=0.2,
        **kwargs,
    ):
        super(DecoderConfig).__init__(**kwargs)
        self.dec_layers = dec_layers
        self.dec_hidden_size = dec_hidden_size
        self.dec_heads = dec_heads
        self.dec_ff_size = dec_ff_size
        self.dec_dropout = dec_dropout

from enum import IntEnum

class QuantizationBits(IntEnum):
    FULL_PRECISION = 32
    HALF_PRECISION = 16
    INT8 = 8
    INT4 = 4

class Config:
    MODEL_NAME: str = "openai-community/gpt2"
    PORT: int = 8000
    MAX_TOKENS: int = 100
    QUANTIZATION_BITS: QuantizationBits = QuantizationBits.FULL_PRECISION
    
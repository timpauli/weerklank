"""Meta's M2M100 translation model, run with CTranslate2.

About twice as fast as with transformers, with the same translations.
"""

from pathlib import Path

from loguru import logger

from echo_lab.config import ConfigError
from echo_lab.llm.base import TranslationService
from echo_lab.registry import StageContext


class M2M100CT2TranslationService(TranslationService):
    """Translates with an M2M100 model converted to CTranslate2.

    The Hugging Face tokenizer still does the tokenization; CTranslate2 runs the model.
    """

    def __init__(
        self,
        *,
        model: str = "facebook/m2m100_418M",
        model_dir: str = "models/m2m100_418M-ct2-float16",
        device: str = "cuda:0",
        compute_type: str = "float16",
        beam_size: int = 4,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        """Load (and, the first time, convert) the model.

        Args:
            model: Hugging Face repo of the M2M100 checkpoint; gives the tokenizer,
                and the weights if ``model_dir`` doesn't exist yet.
            model_dir: Converted CTranslate2 model. Created from ``model`` (stored as
                float16) when missing.
            device: ``"cuda:N"`` or ``"cpu"``.
            compute_type: CTranslate2 compute type, e.g. ``"float16"``,
                ``"int8_float16"`` (smaller, a little faster) or ``"int8"`` (CPU).
            beam_size: Beam search width; 1 is greedy decoding (fastest).
            max_new_tokens: Upper bound on the translation length in tokens.
            **kwargs: ``source_language``/``target_language`` and AIService arguments.
        """
        super().__init__(model=model, **kwargs)
        self._device = device
        self._beam_size = beam_size
        self._max_new_tokens = max_new_tokens
        self._load(model, Path(model_dir), compute_type)

    def _load(self, model: str, model_dir: Path, compute_type: str):
        import ctranslate2
        from transformers import M2M100Tokenizer

        self._tokenizer = M2M100Tokenizer.from_pretrained(model)
        for language in (self.source_language, self.target_language):
            if language not in self._tokenizer.lang_code_to_id:
                raise ConfigError(f"{model} does not support language '{language}'")

        if not (model_dir / "model.bin").is_file():
            from ctranslate2.converters import TransformersConverter

            logger.info(f"Converting {model} to CTranslate2 in {model_dir} (once)")
            TransformersConverter(model).convert(str(model_dir), quantization="float16")

        device, _, index = self._device.partition(":")
        logger.info(f"Loading M2M100 (CTranslate2, {compute_type}) on {self._device}")
        self._translator = ctranslate2.Translator(
            str(model_dir), device=device, device_index=int(index or 0), compute_type=compute_type
        )
        # Loads the CUDA kernels before the first real request.
        self._translate("Hello.")

    def _translate(self, text: str) -> str:
        self._tokenizer.src_lang = self.source_language
        tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
        result = self._translator.translate_batch(
            [tokens],
            target_prefix=[[self._tokenizer.get_lang_token(self.target_language)]],
            beam_size=self._beam_size,
            max_decoding_length=self._max_new_tokens,
        )
        ids = self._tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
        return self._tokenizer.decode(ids, skip_special_tokens=True)


def create(ctx: StageContext):
    """Build the CTranslate2 M2M100 service.

    Options (``llm.m2m100_ct2``): model, model_dir, device, compute_type, beam_size,
    max_new_tokens.
    """
    return M2M100CT2TranslationService(
        source_language=ctx.input_language,
        target_language=ctx.output_language,
        **ctx.options,
    )

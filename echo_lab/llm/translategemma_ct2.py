"""Google's TranslateGemma translation model, run with CTranslate2.

About 4.7x faster than with transformers, with the same translations. The model
is gated: accept its license on Hugging Face and log in (``uv run hf auth login``)
before the first run.
"""

from pathlib import Path

from loguru import logger

from echo_lab.llm.base import TranslationService
from echo_lab.registry import StageContext


class TranslateGemmaCT2TranslationService(TranslationService):
    """Translates with TranslateGemma converted to CTranslate2.

    The Hugging Face processor builds the prompt from the model's own chat
    template and tokenizes it; CTranslate2 runs the model. Only the model's text
    part is converted: translation never uses its image part.
    """

    def __init__(
        self,
        *,
        model: str = "google/translategemma-4b-it",
        model_dir: str = "models/translategemma-4b-it-ct2-bfloat16",
        device: str = "cuda:0",
        compute_type: str = "bfloat16",
        max_new_tokens: int = 512,
        **kwargs,
    ):
        """Load (and, the first time, convert) the model.

        Args:
            model: Hugging Face repo of the TranslateGemma checkpoint; gives the
                prompt template and tokenizer, and the weights if ``model_dir``
                doesn't exist yet.
            model_dir: Converted CTranslate2 model. Created from ``model`` (stored
                as bfloat16, about 7 GB) when missing.
            device: ``"cuda:N"`` or ``"cpu"``.
            compute_type: CTranslate2 compute type. ``"bfloat16"`` is recommended:
                Gemma can overflow in float16, and int8 was no faster in testing.
            max_new_tokens: Upper bound on the translation length in tokens.
            **kwargs: ``source_language``/``target_language`` and AIService arguments.
        """
        super().__init__(model=model, **kwargs)
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._load(model, Path(model_dir), compute_type)

    def _load(self, model: str, model_dir: Path, compute_type: str):
        import ctranslate2
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(model)
        self._tokenizer = self._processor.tokenizer

        if not (model_dir / "model.bin").is_file():
            from ctranslate2.converters import TransformersConverter

            logger.info(f"Converting {model} to CTranslate2 in {model_dir} (once)")
            TransformersConverter(model).convert(str(model_dir), quantization="bfloat16")

        device, _, index = self._device.partition(":")
        logger.info(f"Loading TranslateGemma (CTranslate2, {compute_type}) on {self._device}")
        self._generator = ctranslate2.Generator(
            str(model_dir), device=device, device_index=int(index or 0), compute_type=compute_type
        )
        # Checks that the chat template accepts both language codes, and loads
        # the CUDA kernels before the first real request.
        self._translate("Hello.")

    def _translate(self, text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_language,
                        "target_lang_code": self.target_language,
                        "text": text,
                    }
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        result = self._generator.generate_batch(
            [self._tokenizer.convert_ids_to_tokens(ids)],
            max_length=self._max_new_tokens,
            sampling_topk=1,  # greedy, like the model card
            include_prompt_in_result=False,
            end_token=["<end_of_turn>", "<eos>"],
        )
        return self._tokenizer.decode(result[0].sequences_ids[0], skip_special_tokens=True)


def create(ctx: StageContext):
    """Build the CTranslate2 TranslateGemma service.

    Options (``llm.translategemma_ct2``): model, model_dir, device, compute_type,
    max_new_tokens.
    """
    return TranslateGemmaCT2TranslationService(
        source_language=ctx.input_language,
        target_language=ctx.output_language,
        **ctx.options,
    )

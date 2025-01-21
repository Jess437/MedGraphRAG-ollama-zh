from typing import Any, List, Optional

from langchain_core.language_models import BaseLanguageModel

from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)

from langchain_helper.output_parser import JsonKeyOutputFunctionsParser


def _get_extraction_function(entity_schema: dict) -> dict:
    return {
        "name": "information_extraction",
        "description": "Extracts the relevant information from the passage.",
        "parameters": {
            "type": "object",
            "properties": {
                "info": {"type": "array", "items": _convert_schema(entity_schema)}
            },
            "required": ["info"],
        },
    }

_EXTRACTION_TEMPLATE ="""請從以下段落中提取並保存提到的相關實體及其屬性。
僅提取在 'information_extraction' 函數中提到的屬性。
如果某個屬性沒有出現或不是該函數所需要的參數，請勿將其包含在輸出中。

段落：{input}
"""  # noqa: E501

def create_extraction_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    tags: Optional[List[str]] = None,
    verbose: bool = False,
) -> Chain:
    """Creates a chain that extracts information from a passage.

    Args:
        schema: The schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console. Defaults to the global `verbose` value,
            accessible via `langchain.globals.get_verbose()`.

    Returns:
        Chain that can be used to extract information from a passage.
    """
    function = _get_extraction_function(schema)
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = JsonKeyOutputFunctionsParser(key_name="info")
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        tags=tags,
        verbose=verbose,
    )
    return chain
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


EVIDENCE_BASED_ASSESSMENT_INSTRUCTION = """IMPORTANT: Answer strictly based on the provided context above. Do NOT use external knowledge or information not present in the context.

Before answering, audit the provided context against the question. Provide a concise evidence analysis that can be checked:
- Point 1: Quote the exact sentence(s) from the context that directly answer the question, focusing on content that matches the question's key terms.
- Point 2: Identify any additional evidence, constraints, dates, entities, numbers, or multi-hop links needed for the answer.
- Point 3: State whether any key information is missing or conflicting.

If the context is INSUFFICIENT (missing key facts, conflicting information, or would require guessing), set "sufficient" to false, list the missing information, and set "answer" to "Not mentioned". Do NOT guess or fabricate.

If the context is SUFFICIENT, set "sufficient" to true and provide a complete answer in the "answer" field. Include all relevant details (dates, ranges, names) rather than oversimplifying.

Respond ONLY as a JSON object in the following format:
{
  "sufficient": true/false,
  "evidence_analysis": [
    "Point 1: [Quote] ...",
    "Point 2: ...",
    "Point 3: ..."
  ],
  "missing_info": [],
  "answer": "<final answer or Not mentioned>",
  "reasoning": "<one short sentence summarizing why the answer is supported or why it is insufficient>"
}"""


@dataclass
class StandardQA:
    """Standardized single question-answer pair"""
    question: str
    gold_answers: List[str]
    evidence: List[str] = field(default_factory=list)
    category: Optional[Union[int, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardSample:
    """Standardized sample containing document content and corresponding QA list"""
    sample_id: str
    qa_pairs: List[StandardQA]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class StandardDoc:
    """Standardized sampleid to doc_path mapping structure"""
    sample_id:str
    doc_paths:list[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """Base class for all dataset adapters"""
    
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        self.logger = get_logger()

    @abstractmethod
    def data_prepare(self, doc_dir:str) -> List[StandardDoc]:
        """
        Data preparation.
        1. Convert dataset format to store-friendly files
        2. Return converted (or unconverted) file paths
        
        Returns:
            List[StandardDoc]: Array of file paths expected by the active store
        """
        pass

    @abstractmethod
    def load_and_transform(self) -> List[StandardSample]:
        """
        Read raw files and convert to standard format list.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        Build final prompt to send to LLM based on retrieved context and QA pair.
        
        Returns:
            - full_prompt (str): Complete prompt string
            - meta (Dict): Metadata to pass to post-processing function (e.g., option mapping for multiple choice)
        """
        pass

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """
        Post-process raw LLM output (default implementation only strips whitespace).
        """
        return raw_answer.strip()

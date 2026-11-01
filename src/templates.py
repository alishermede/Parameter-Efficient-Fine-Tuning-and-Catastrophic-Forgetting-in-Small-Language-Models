"""Prompt templates shared by every method, training and development evaluation.

All three adaptation methods see exactly the same text. Base (non-instruct) checkpoints
are used, so no chat template is applied.
"""
from __future__ import annotations

INSTR_HEADER = "### Instruction:\n"
RESP_HEADER = "\n\n### Response:\n"
ANSWER_PREFIX = "The answer is"
LETTERS = "ABCDE"

DOMAIN_DESCRIPTION = {
    "medmcqa": "medicine",
    "casehold": "U.S. case law",
}

CASEHOLD_INSTRUCTION = (
    "Choose the holding that correctly fills the <HOLDING> placeholder "
    "in the following excerpt from a U.S. court opinion."
)


def mcq_instruction(domain: str, question: str, options: list[str]) -> str:
    if domain == "casehold":
        head = CASEHOLD_INSTRUCTION
    else:
        head = f"Answer the following multiple-choice question about {DOMAIN_DESCRIPTION[domain]}."
    lines = [head, "", question.strip()]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {str(opt).strip()}")
    return "\n".join(lines)


def mcq_prompt(domain: str, question: str, options: list[str]) -> str:
    return INSTR_HEADER + mcq_instruction(domain, question, options) + RESP_HEADER


def mcq_response(options: list[str], label: int, explanation: str | None = None) -> str:
    text = f"{ANSWER_PREFIX} {LETTERS[label]}. {str(options[label]).strip()}"
    if explanation and explanation.strip():
        text += f"\nExplanation: {explanation.strip()}"
    return text


def code_prompt(instruction: str, inp: str | None = None) -> str:
    text = instruction.strip()
    if inp and inp.strip():
        text += "\n\nInput:\n" + inp.strip()
    return INSTR_HEADER + text + RESP_HEADER


def mbpp_dev_prompt(text: str, tests: list[str]) -> str:
    instruction = (
        f"{text.strip()}\nWrite a Python function. Your code should pass these tests:\n" + "\n".join(tests)
    )
    return code_prompt(instruction)

from frontierguard.traces.segment import segment_reasoning
from frontierguard.traces.verify import extract_final_answer, verify_math_answer


def test_segment_blank_lines_and_numbered_steps():
    text = "1. Compute x = 2 + 3.\n\n2. Therefore x=5.\n\nFinal answer: 5"
    steps = segment_reasoning(text)
    assert len(steps) == 3
    assert steps[0].text.startswith("1.")
    assert steps[-1].char_end == len(text)


def test_extract_boxed_and_verify_fraction():
    assert extract_final_answer(r"work \boxed{1/2}") == "1/2"
    assert verify_math_answer("1/2", "0.5")
    assert verify_math_answer("50%", "0.5")
    assert not verify_math_answer("0.6", "0.5")

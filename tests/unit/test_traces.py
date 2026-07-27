from frontierguard.traces.segment import segment_reasoning
from frontierguard.traces.verify import extract_final_answer, verify_math_answer


def test_segment_blank_lines_and_numbered_steps():
    text = "1. Compute x = 2 + 3.\n\n2. Therefore x=5.\n\nFinal answer: 5"
    steps = segment_reasoning(text)
    assert len(steps) == 3
    assert steps[0].text.startswith("1.")
    assert steps[-1].char_end == len(text)
    assert steps[0].eligible
    assert not steps[-1].eligible
    assert steps[-1].kind == "answer"


def test_segment_excludes_think_boundary_and_presentation():
    text = (
        "Compute 31 + 19 = 50.\n\n"
        "Then 80% of 50 is 40.\n"
        "</think>\n\n"
        "**Solution:**\n\n"
        "The answer is 40."
    )
    steps = segment_reasoning(text)
    eligible = [step for step in steps if step.eligible]
    presentation = [step for step in steps if step.phase == "presentation"]

    assert [step.text for step in eligible] == [
        "Compute 31 + 19 = 50.",
        "Then 80% of 50 is 40.",
    ]
    assert any(step.text == "</think>" and step.kind == "format" for step in steps)
    assert all(not step.eligible for step in presentation)
    assert any(step.text == "**Solution:**" for step in presentation)


def test_numbered_markdown_heading_is_not_a_reasoning_candidate():
    steps = segment_reasoning(
        "1. **Determine the Total Number of Days:**\n\n"
        "March has 31 days and April contributes 19, so 31 + 19 = 50."
    )
    assert steps[0].kind == "format"
    assert not steps[0].eligible
    assert steps[1].eligible


def test_solution_label_with_substantive_reasoning_is_kept_without_think_tags():
    steps = segment_reasoning("Solution: We compute 31 + 19 = 50.")
    assert len(steps) == 1
    assert steps[0].kind == "content"
    assert steps[0].eligible


def test_extract_boxed_and_verify_fraction():
    assert extract_final_answer(r"work \boxed{1/2}") == "1/2"
    assert verify_math_answer("1/2", "0.5")
    assert verify_math_answer("50%", "0.5")
    assert not verify_math_answer("0.6", "0.5")

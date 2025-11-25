import json
from functools import partial
from multiprocessing import Pool

import dspy
import wikipedia
from dotenv import load_dotenv

class FutureTimeline(dspy.Signature):
    """Generate a realistic chronological timeline related to the scenario or topic from the starting date to the foreseeable future"""

    timeline_scenario: str = dspy.InputField()
    contexts: list[str] = dspy.InputField()
    current_date: str = dspy.InputField()
    simulated_timeline: str = dspy.OutputField()


class TimelineImplication(dspy.Signature):
    """Given a simulated future timeline and a question related to the timeline, return whether time timeline as stated implies the answer to the question being true"""

    timeline: str = dspy.InputField()
    question_to_answer: str = dspy.InputField()
    implied_answer: bool = dspy.OutputField(
        desc="The answer to the question, given the timeline"
    )


def run_model(model, contexts):
    dspy.configure(lm=dspy.LM(model))
    temps = [0.1, 0.3, 0.5, 0.7, 0.9]
    rval = []
    for rollout_id, temp in enumerate(temps):
        try:
            timeline_prediction = dspy.Predict(
                FutureTimeline, rollout_id=rollout_id, temperature=temp
            )
            question_answering = dspy.ChainOfThought(TimelineImplication, temperature=0)

            final_question = "Russia x Ukraine ceasefire by end of 2026?"

            timeline = timeline_prediction(
                timeline_scenario="Russia/Ukraine conflict",
                contexts=contexts,
                current_date="2025-11-25",
            )
            timeline_implication = question_answering(
                timeline=timeline.simulated_timeline, question_to_answer=final_question
            )
            rval.append(
                {
                    "model": model,
                    "rollout_id": rollout_id,
                    "temp": temp,
                    "simulated_timeline": timeline.simulated_timeline,
                    "choice": timeline_implication.implied_answer,
                }
            )
        except:
            rval.append({"model": model, "rollout_id": rollout_id, "error": True})

    return rval


def main():
    models = [
        "openrouter/anthropic/claude-opus-4.5",
        "openrouter/openrouter/bert-nebulon-alpha",
        "openrouter/x-ai/grok-4.1-fast:free",
        "openrouter/google/gemini-3-pro-preview",
        "openrouter/openai/gpt-5.1",
    ]
    context_page_titles = [
        "Timeline of the Russo-Ukrainian war (1 September 2025 – present)",
        "Peace negotiations in the Russo-Ukrainian war (2022–present)",
    ]
    contexts = []
    for page_title in context_page_titles:
        page = wikipedia.page(page_title)
        context = page.title + "\n" + page.content
        contexts.append(context)
    with Pool(processes=len(models)) as pool:
        processed = pool.map(partial(run_model, contexts=contexts), models)
        for model_results in processed:
            for result in model_results:
                print(json.dumps(result))


if __name__ == "__main__":
    load_dotenv()
    main()

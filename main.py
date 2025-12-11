import json
import logging

from dotenv import load_dotenv
from fire import Fire

from src.generate_future_timeline import generate_future_timeline
from src.generate_timeline_to_now import generate_timeline_to_now


def generate_future():
    scenario = "Russia/Ukraine conflict"
    context_page_titles = [
        "Timeline of the Russo-Ukrainian war (1 September 2025 – present)",
        "Peace negotiations in the Russo-Ukrainian war (2022–present)",
    ]
    current_date = "2025-11-25"
    final_question = "Russia x Ukraine ceasefire by end of 2026?"
    results = generate_future_timeline(
        scenario, context_page_titles, current_date, final_question
    )
    with open("output_future_timeline.json", "w") as o:
        print(json.dumps(results), file=o)


def generate_to_now():
    output = generate_timeline_to_now("Russia/Ukraine conflict", "2025-11-25")
    with open("output_current_timeline.json", "w") as o:
        print(json.dumps(output), file=o)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    Fire()

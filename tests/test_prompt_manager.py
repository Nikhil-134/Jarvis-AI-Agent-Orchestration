"""Tests for prompt template rendering."""

from llm import PromptManager


def test_prompt_manager_renders_registered_template() -> None:
    manager = PromptManager({"demo": "Hello $name"})

    assert manager.render("demo", name="Jarvis") == "Hello Jarvis"


def test_prompt_manager_allows_template_registration() -> None:
    manager = PromptManager()
    manager.register_template("task", "Task: $task")

    assert manager.render("task", task="plan") == "Task: plan"

"""Tests for prompt template rendering."""

from llm import PromptManager


def test_prompt_manager_renders_registered_template() -> None:
    """PromptManager should render templates with supplied variables."""
    manager = PromptManager({"demo": "Hello $name"})

    assert manager.render("demo", name="Jarvis") == "Hello Jarvis"


def test_prompt_manager_allows_template_registration() -> None:
    """PromptManager should allow runtime template registration."""
    manager = PromptManager()
    manager.register_template("task", "Task: $task")

    assert manager.render("task", task="plan") == "Task: plan"

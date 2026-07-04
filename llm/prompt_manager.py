"""Prompt template management."""

from string import Template


class PromptManager:
    """Stores and renders named prompt templates."""

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        """Initialize prompt templates."""
        self._templates = {
            "planner": (
                "Create a concise execution plan for this goal. "
                "Return actionable steps only.\n\nGoal: $goal"
            )
        }
        self._templates.update(templates or {})

    def register_template(self, name: str, template: str) -> None:
        """Register or replace a named template."""
        self._templates[name] = template

    def render(self, template_name: str, **variables: object) -> str:
        """Render a named template with variables."""
        if template_name not in self._templates:
            raise KeyError(f"Prompt template is not registered: {template_name}")
        return Template(self._templates[template_name]).safe_substitute(
            {key: str(value) for key, value in variables.items()}
        )


if __name__ == "__main__":
    print(PromptManager().render("planner", goal="demo"))

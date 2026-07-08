"""Prompt template management."""

from string import Template


class PromptManager:
    """Stores and renders named prompt templates."""

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        """Initialize prompt templates."""
        self._templates = {
            "responder": (
                "You are Jarvis, a helpful AI assistant. Respond to the user "
                "naturally and conversationally.\n\n"
                "Relevant memory context:\n$memory_context\n\n"
                "$tool_results\n"
                "User: $goal\nAssistant:"
            ),
            "decomposer": (
                "Given the user goal, identify which specialist task types "
                "are needed. Available specialists:\n$specialist_list\n\n"
                "Goal: $goal\n\n"
                "Return ONLY a comma-separated list of the most relevant "
                "task_type values, nothing else."
            ),
            "merger": (
                "Combine the following results from multiple specialist "
                "agents into one coherent, natural response.\n\n"
                "Original request: $goal\n\n"
                "Agent results:\n$agent_results\n\n"
                "Provide a unified, conversational response that covers "
                "all key information without redundancy."
            ),
        }
        self._templates.update(templates or {})

    def register_template(self, name: str, template: str) -> None:
        """Register or replace a named template."""
        self._templates[name] = template

    def render(self, template_name: str, **variables: object) -> str:
        """Render a named template with variables.

        Any placeholder not supplied by *variables* is substituted with an
        empty string rather than leaking a literal ``$name`` into the prompt.
        Leaking ``$tool_results`` previously confused small local models and
        produced empty or malformed responses.
        """
        if template_name not in self._templates:
            raise KeyError(f"Prompt template is not registered: {template_name}")

        template = Template(self._templates[template_name])
        rendered = {key: str(value) for key, value in variables.items()}
        # Fill every remaining identifier with "" so no raw $placeholder leaks.
        for identifier in template.get_identifiers():
            rendered.setdefault(identifier, "")
        return template.substitute(rendered)


if __name__ == "__main__":
    print(PromptManager().render("planner", goal="demo"))

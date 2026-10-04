"""Keep original full budgets; --hypothesis-profile=dev caps each at twenty."""

from hypothesis import settings


def property_settings(*, max_examples: int, **kwargs):
    # The loaded public settings object identifies dev without private Hypothesis
    # attributes. Full uses each existing decorator's original example budget.
    if settings.default == settings.get_profile("dev"):
        max_examples = min(max_examples, 20)
    return settings(max_examples=max_examples, **kwargs)

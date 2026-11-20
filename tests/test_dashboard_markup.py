"""Static dashboard markup must stay well nested.

Regression guard for the extra closing ``</div>`` that escaped the rule modal
body and pushed the dialog footer out of its dialog, breaking the modal layout.
"""

from html.parser import HTMLParser
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parents[1] / "llmproxy" / "static" / "index.html"

VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

RULE_LIMIT_INPUTS = {"rule-form-context", "rule-form-max-tokens"}


class _MarkupChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.errors: list[tuple[str, str | None, tuple[int, int]]] = []
        self.modal_footer_parents: list[str] = []
        self.limit_input_ancestors: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        if attr_map.get("id") in RULE_LIMIT_INPUTS:
            self.limit_input_ancestors[attr_map["id"]] = [
                parent_attrs.get("class", "") for _tag, parent_attrs in self.stack
            ]
        if tag in VOID_ELEMENTS:
            return
        self.stack.append((tag, attr_map))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_ELEMENTS:
            return
        if not self.stack or self.stack[-1][0] != tag:
            self.errors.append((tag, self.stack[-1][0] if self.stack else None, self.getpos()))
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    del self.stack[index:]
                    return
            return
        _tag, attr_map = self.stack.pop()
        classes = attr_map.get("class", "").split()
        if "modal-footer" in classes:
            parent = self.stack[-1][1].get("class", "") if self.stack else ""
            self.modal_footer_parents.append(parent)


def _checker() -> _MarkupChecker:
    checker = _MarkupChecker()
    checker.feed(INDEX_PATH.read_text(encoding="utf-8"))
    checker.close()
    return checker


def test_dashboard_markup_is_balanced():
    checker = _checker()
    assert checker.errors == []
    assert checker.stack == []


def test_modal_footers_stay_inside_modal_dialogs():
    checker = _checker()
    assert checker.modal_footer_parents, "expected modal footers in the dashboard"
    for parent in checker.modal_footer_parents:
        assert "modal-dialog" in parent.split()


def test_rule_limit_inputs_live_inside_the_rule_modal_body():
    checker = _checker()
    for input_id in RULE_LIMIT_INPUTS:
        ancestors = checker.limit_input_ancestors.get(input_id, [])
        assert ancestors, f"{input_id} not found in index.html"
        assert any("modal-body" in classes.split() for classes in ancestors)

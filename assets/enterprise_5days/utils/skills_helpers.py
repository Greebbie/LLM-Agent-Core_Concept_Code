"""Anthropic Skills helpers — minimal teaching wrappers.

Anthropic Skills are folder-based packageable expertise:
    skill_dir/
        SKILL.md            # YAML frontmatter + body (mandatory)
        helper.py           # optional helper scripts
        reference/          # optional progressive-disclosure docs
            *.md / *.json / *.jsonl

Frontmatter (YAML) supports:
    name: short-name              # required
    description: one-line trigger # required (LLM uses this to select)
    allowed-tools: [tool1, tool2] # optional (limit Skill's tool access)
    model: claude-3-5-sonnet      # optional (which model to use)
    version: "0.1"                # optional

This module provides parsing/validation/discovery + a small LLM-routing demo.
For real production, use the official Claude Agent SDK or Claude Code's
built-in skill loader.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ============================================================
# Skill dataclass
# ============================================================
@dataclass
class Skill:
    name: str
    description: str
    body: str
    skill_dir: Path
    frontmatter: dict = field(default_factory=dict)
    helper_files: list[Path] = field(default_factory=list)
    reference_files: list[Path] = field(default_factory=list)

    @property
    def allowed_tools(self) -> list[str]:
        return self.frontmatter.get("allowed-tools", []) or []

    @property
    def model(self) -> Optional[str]:
        return self.frontmatter.get("model")

    @property
    def version(self) -> str:
        return str(self.frontmatter.get("version", "0.1"))


# ============================================================
# Parse / validate
# ============================================================
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_skill_md(skill_md_path: str | Path) -> tuple[dict, str]:
    """Parse a SKILL.md file → (frontmatter_dict, body_str).

    Frontmatter is YAML between two `---` lines. Body is everything after.
    """
    text = Path(skill_md_path).read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{skill_md_path}: missing or malformed YAML frontmatter")
    raw_yaml, body = m.group(1), m.group(2)
    fm = _parse_simple_yaml(raw_yaml)
    return fm, body.strip()


def _parse_simple_yaml(yaml_text: str) -> dict:
    """Minimal YAML parser: supports key: value and key: [a, b, c]. No nested."""
    out = {}
    for line in yaml_text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # List
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            out[key] = [s.strip().strip('"').strip("'") for s in inner.split(",") if s.strip()] if inner else []
        # String
        elif value:
            out[key] = value.strip('"').strip("'")
        else:
            out[key] = ""
    return out


def validate_skill(skill_dir: str | Path) -> dict:
    """Check skill folder structure + required fields.
    Returns: {"ok": bool, "errors": [...], "warnings": [...]}
    """
    skill_dir = Path(skill_dir)
    errors, warnings = [], []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return {"ok": False, "errors": [f"Missing SKILL.md in {skill_dir}"], "warnings": []}
    try:
        fm, body = parse_skill_md(skill_md)
    except Exception as e:
        return {"ok": False, "errors": [f"Cannot parse SKILL.md: {e}"], "warnings": []}
    # Required fields
    for required in ("name", "description"):
        if required not in fm or not fm[required]:
            errors.append(f"Frontmatter missing required field: {required}")
    # Description quality
    desc = fm.get("description", "")
    if len(desc) < 20:
        warnings.append(f"description too short ({len(desc)} chars); LLM hard to route well")
    if len(desc) > 300:
        warnings.append(f"description too long ({len(desc)} chars); should be 1-2 sentences")
    # Body
    if len(body) < 50:
        warnings.append("body very short; may not give Claude enough guidance")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


# ============================================================
# Discovery
# ============================================================
def discover_skills(skills_root: str | Path) -> list[Skill]:
    """Walk a directory of skills, return list of loaded Skill objects.
    Skips invalid skills (logs warning)."""
    skills_root = Path(skills_root)
    skills = []
    if not skills_root.exists():
        return skills
    for item in sorted(skills_root.iterdir()):
        if not item.is_dir():
            continue
        skill_md = item / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            fm, body = parse_skill_md(skill_md)
            helpers = [p for p in item.iterdir() if p.suffix == ".py"]
            ref_dir = item / "reference"
            refs = sorted(ref_dir.iterdir()) if ref_dir.exists() else []
            skills.append(Skill(
                name=fm.get("name", item.name),
                description=fm.get("description", ""),
                body=body,
                skill_dir=item,
                frontmatter=fm,
                helper_files=helpers,
                reference_files=refs,
            ))
        except Exception as e:
            print(f"⚠ Skipping {item}: {e}")
    return skills


# ============================================================
# LLM routing demo: pick the right skill for a query
# ============================================================
def match_skill_for_query(query: str, skills: list[Skill], llm) -> Optional[Skill]:
    """Use LLM to pick the most relevant skill (or None if no good match).

    This mimics how Claude Code / Claude.ai uses each skill's `description`
    field to decide which skill to load. Real Claude does it automatically;
    here we make it explicit for teaching.
    """
    if not skills:
        return None
    catalog = "\n".join(
        f"- [{i}] {s.name}: {s.description}" for i, s in enumerate(skills)
    )
    prompt = f"""你需要为用户的查询挑选最合适的 skill（如有）。

可用 skills:
{catalog}

用户查询: {query}

如果有 skill 明显匹配，输出该 skill 的索引数字（0/1/2...）。
如果都不匹配，输出 NONE。
只输出索引数字或 NONE，无需理由。"""
    raw = llm.generate(prompt, temperature=0.0).strip()
    if "NONE" in raw.upper():
        return None
    # Extract first integer
    m = re.search(r"\d+", raw)
    if m:
        idx = int(m.group())
        if 0 <= idx < len(skills):
            return skills[idx]
    return None


# ============================================================
# Progressive disclosure demo
# ============================================================
def load_skill_progressive(skill: Skill, query: str, llm) -> dict:
    """Demonstrate progressive disclosure:
    1. Always load: name, description (cheap — already known from discovery)
    2. On match: load body
    3. On demand: load reference/* files individually based on query

    Returns dict with what was loaded, mimicking Claude's loading pattern.
    """
    loaded = {
        "name": skill.name,
        "description": skill.description,
        "body": skill.body,
        "references_loaded": [],
        "tokens_estimate": len(skill.body) // 4,
    }
    if not skill.reference_files:
        return loaded
    # Decide which references to load
    ref_catalog = "\n".join(f"- {p.name}" for p in skill.reference_files)
    prompt = f"""用户查询: {query}
Skill body: {skill.body[:500]}...

可选参考文档:
{ref_catalog}

请输出**逗号分隔**的需要加载的文件名（最多 2 个）。如不需要加载任何，输出 NONE。"""
    raw = llm.generate(prompt, temperature=0.0).strip()
    if "NONE" in raw.upper():
        return loaded
    wanted = {n.strip() for n in raw.split(",") if n.strip()}
    for ref in skill.reference_files:
        if ref.name in wanted:
            content = ref.read_text(encoding="utf-8")
            loaded["references_loaded"].append({"name": ref.name, "content": content[:1000]})
            loaded["tokens_estimate"] += len(content) // 4
    return loaded

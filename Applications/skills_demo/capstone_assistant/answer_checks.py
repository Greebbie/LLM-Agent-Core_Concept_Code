"""按题目要求核对答案要点。同组接受任一写法，不同组都要命中。"""
import re


def matches_facts(answer, case):
    # 检索到正确资料，和用它回答了问题，是两件事。
    answer = answer.partition("\n\n依据文档:")[0]
    if answer.startswith("[MCP 调用失败]"):
        return False
    if answer.startswith("调用 MCP tool "):
        answer = answer.partition(" 得到: ")[2]
    groups = case.get("required_facts") or [case["expected_keywords"]]

    def contains(term):
        pattern = re.escape(term)
        if term.isascii() and term.isalpha():
            pattern = r"(?<![a-z])" + pattern + r"(?![a-z])"
        if term[0].isdecimal():
            pattern = r"(?<![\d.\-])" + pattern
        if term[-1].isdecimal():
            pattern += r"(?!\d|\.\d*[1-9])"
        return re.search(pattern, answer, flags=re.IGNORECASE) is not None

    return all(any(contains(term) for term in alternatives) for alternatives in groups)

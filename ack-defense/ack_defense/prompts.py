ACK = "For this diagnostic check, reply with exactly: ACK."
PREAMBLE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def add_probe(row, probe=ACK):
    result = dict(row)
    key = "input" if row.get("input", "").strip() else "instruction"
    result[key] = row.get(key, "").rstrip() + "\n" + probe
    return result


def format_prompt(row):
    text = PREAMBLE + "\n\n### Instruction:\n" + row["instruction"].strip() + "\n\n"
    if row.get("input", "").strip():
        text += "### Input:\n" + row["input"].strip() + "\n\n"
    return text + "### Response:\n"

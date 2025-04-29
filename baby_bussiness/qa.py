# qa.py
import re
import difflib
from data_loader import load_data
from model import SimpleGenAI

# map common query terms to business‐level fields
FIELD_SYNONYMS = {
    'employee':         'team size',
    'employees':        'team size',
    'headcount':        'team size',
    'sale':             'total sales',
    'sales':            'total sales',
    'expense':          'total expenses',
    'expenses':         'total expenses',
    'productivity':     'average productivity',
    'profit':           'total profit',
    'revenue':          'total revenue',
    'margin':           'net profit margin',
}

def resolve_field(question, biz_fields, emp_fields):
    """
    First try exact or synonym matches on business‐level stats.
    Then try fuzzy match across keys.
    Finally if it mentions an employee name, do fuzzy match
    within that employee's fields.
    """
    tokens = re.findall(r'\w+', question.lower())

    # 1) synonyms
    for tok in tokens:
        if tok in FIELD_SYNONYMS:
            key = FIELD_SYNONYMS[tok]
            if key in biz_fields:
                return key, biz_fields[key]

    # 2) exact token match on biz fields
    for tok in tokens:
        if tok in biz_fields:
            return tok, biz_fields[tok]

    # 3) fuzzy match on biz fields
    choices = list(biz_fields.keys())
    match = difflib.get_close_matches(question.lower(), choices, n=1, cutoff=0.5)
    if match:
        return match[0], biz_fields[match[0]]

    # 4) check for employee name + fuzzy field
    for emp_name in {e.split()[0].lower() for e in emp_fields}:
        if emp_name in tokens:
            # gather fields for this person
            person_keys = [k for k in emp_fields if k.lower().startswith(emp_name)]
            match = difflib.get_close_matches(question.lower(), person_keys, n=1, cutoff=0.4)
            if match:
                return match[0], emp_fields[match[0]]

    return None, None


def main():
    data = load_data()
    # prepare business‐level aggregates
    biz = {
        'team size':          data.get('team_size', 0),
        'total sales':        sum(emp['sales'] for emp in data['employees']),
        'total expenses':     sum(emp['expenses'] for emp in data['employees']),
        'total revenue':      sum(emp['revenue'] for emp in data['employees']),
        'total profit':       sum(emp['profit'] for emp in data['employees']),
        'average productivity': round(
                                sum(emp['productivity'] for emp in data['employees'])
                                / len(data['employees']), 2),
        'net profit margin':  round(
                                (sum(emp['profit'] for emp in data['employees'])
                                 / sum(emp['revenue'] for emp in data['employees'])) 
                                if data['employees'] else 0, 2),
    }

    # flatten per‐employee stats into keys like "john doe sales"
    emps = {}
    for emp in data['employees']:
        name = emp['name']
        for k, v in emp.items():
            if k not in ('id','name'):
                emps[f"{name.lower()} {k.replace('_',' ')}"] = v

    ai = SimpleGenAI(d_model=32)

    print("=== SimpleGenAI for Business Stats ===")
    print("Type your question (or 'quit' to exit):")

    while True:
        q = input(">> ")
        if q.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # 1) Try rule‐based resolution first
        field, value = resolve_field(q, biz, emps)
        if field:
            print("\n--- RULE‐BASED ANSWER ---")
            print(f"Your {field} is {value}.")
            continue

        # 2) Fallback to vector matching
        print("\n--- DEBUG STEPS ---")
        ans = ai.answer(q, {**biz, **emps})
        print("\n--- ANSWER ---")
        print(ans)
        print("-------------------\n")


if __name__ == "__main__":
    main()

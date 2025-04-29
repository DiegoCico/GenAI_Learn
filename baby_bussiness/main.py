#!/usr/bin/env python3
import json
import os

# Load data from JSON file
DATA_FILE = os.path.join(os.path.dirname(__file__), 'business_data.json')
with open(DATA_FILE, 'r') as f:
    data = json.load(f)

employees = data["employees"]
team_size = data["team_size"]
quarter = data.get("quarter", "")

metrics = [
    "productivity",
    "sales",
    "expenses",
    "projects_completed",
    "client_satisfaction",
    "growth_percentage",
    "revenue",
    "profit",
]

def list_employees():
    return ", ".join(e["name"] for e in employees)

def sum_metric(metric):
    return sum(e.get(metric, 0) for e in employees)

def avg_metric(metric):
    return sum_metric(metric) / len(employees)

def max_employee(metric):
    e = max(employees, key=lambda x: x.get(metric, 0))
    return e["name"], e.get(metric, 0)

def min_employee(metric):
    e = min(employees, key=lambda x: x.get(metric, 0))
    return e["name"], e.get(metric, 0)

def get_employee_by_name(q):
    for e in employees:
        if e["name"].lower() in q:
            return e
    return None

def answer_question(q):
    ql = q.lower().strip()
    if ql in ("quit", "exit", "q"):
        return None

    if "quarter" in ql:
        return f"The current reporting period is {quarter}."

    if "team size" in ql or "number of team" in ql:
        return f"Your team currently has {team_size} members."

    if "list" in ql and "employee" in ql:
        names = list_employees()
        return f"The team consists of the following employees: {names}."

    for m in metrics:
        label = m.replace("_", " ")
        if label in ql or m in ql:
            if "average" in ql or "avg" in ql:
                val = avg_metric(m)
                return f"The average {label} across the team is {val:.2f}."
            if "total" in ql or "sum" in ql:
                val = sum_metric(m)
                return f"The total {label} is {val:.2f}."
            if any(k in ql for k in ("highest", "max", "maximum")):
                name, val = max_employee(m)
                return f"The highest {label} is {val} achieved by {name}."
            if any(k in ql for k in ("lowest", "min", "minimum")):
                name, val = min_employee(m)
                return f"The lowest {label} is {val} by {name}."
            emp = get_employee_by_name(ql)
            if emp:
                val = emp.get(m, "N/A")
                return f"{emp['name']} has a {label} of {val}."
            details = "; ".join(f"{e['name']}={e.get(m)}" for e in employees)
            return f"{label.title()} by employee: {details}."

    return (
        "I’m sorry, I did not understand your question. "
        "You can ask about team size, quarter, list of employees, or metrics "
        "like average sales, highest profit, etc."
    )

def main():
    print(
        "Welcome to your Business GenAI CLI.\n"
        "You can ask me questions about team size, quarter, employees,\n"
        "or metrics like average sales, highest profit, etc.\n"
        "Type 'quit' to exit.\n"
    )
    while True:
        q = input("What would you like to know? ")
        if not q.strip():
            continue
        ans = answer_question(q)
        if ans is None:
            print("Goodbye!")
            break
        print(ans + "\n")

if __name__ == "__main__":
    main()

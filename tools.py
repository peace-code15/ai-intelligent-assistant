import json
import os
from datetime import datetime
from langchain_core.tools import tool

TASKS_FILE = "tasks.json"

def load_tasks() -> list:
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_tasks(tasks: list):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

@tool
def create_action_item(title: str, priority: str = "Medium", due_date: str = "Not specified") -> str:
    """
    Creates and saves a new task, reminder, or action item when the user asks to schedule,
    remind, log, or track something.
    """
    tasks = load_tasks()
    new_task = {
        "id": len(tasks) + 1,
        "title": title,
        "priority": priority.capitalize(),
        "due_date": due_date,
        "status": "Pending",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    tasks.append(new_task)
    save_tasks(tasks)
    return f"Successfully logged task #{new_task['id']}: '{title}' (Priority: {new_task['priority']}, Due: {due_date})."

@tool
def complete_action_item(task_id: int) -> str:
    """Marks an existing task as completed using its task ID number."""
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = "Completed"
            save_tasks(tasks)
            return f"Task #{task_id} ('{task['title']}') marked as Completed."
    return f"No task found with ID #{task_id}."

AVAILABLE_TOOLS = [create_action_item, complete_action_item]
TOOL_MAP = {t.name: t for t in AVAILABLE_TOOLS}
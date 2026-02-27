import threading

running_tasks: dict[str, list] = {}
tasks_lock = threading.Lock()

# aml - Machine learning library.
# Task management module.
# Author: Julian Ailan
# ===================================

"""Provides a class for managing pending tasks."""

# Built-in modules.
import os
import json
import time


class PendingTasks(object):
    """Manages the creation, modification and status of pending tasks."""

    def __init__(self):
        """Tasks manager constructor"""
        self.dir_path = (os.path.dirname(os.path.realpath(__file__))[:-5] +
                         "tasks/pending.json")
        with open(self.dir_path) as pending_tasks:
            self.tasks = json.load(pending_tasks)
            self.next_task_id = self.tasks["next_task_id"]
            self.completed_tasks = self.tasks["completed"]
            self.working_on = self.tasks["working_on"]
            self.not_started = self.tasks["not_started"]
        self.unsaved_changes = False

    def create_task(self, task, depends_from, priority):
        """Creates a new pending task.

        Args:
            task: string that contains a verbose explanation of the new task.
            depends_from: list that contains the task ids that should be first
            compeleted before.
            priority: associated integer value.

        Returns:
            No data is returned.
        """
        new_task = {
            "task": task,
            "created_at": time.strftime("%Y/%m/%d"),
            "depends_from": depends_from,
            "priority": priority,
            "completed": "false"
        }
        task_id = self.next_task_id
        self.next_task_id += 1
        self.not_started[str(task_id)] = new_task
        self.unsaved_changes = True
        print("New task with id #{} added.".format(task_id))

    def modify_task(self, task_id, **field_to_update_and_value):
        """Modifies fields in not started tasks.

        Args:
            task_id: identifier of task to update.
            field_to_update_and_value: key-value arguments based on dictionary
            keys.

        Returns:
            No data is returned.
        """
        if isinstance(task_id, int):
            task_id = str(task_id)
        try:
            if field_to_update_and_value:
                task_to_update = self.not_started[task_id]
                for field, value in field_to_update_and_value.items():
                    task_to_update[field] = value
                task_to_update["modified"] = time.strftime("%Y/%m/%d")
                self.not_started[task_id] = task_to_update
                self.unsaved_changes = True
            else:
                print "No fields selected to update."
        except KeyError:
            print("There is no pending task with such id. Please check the" +
                  " id number of the task to modify again.")

    def start_task(self, task_id):
        """Marks a pending task as started.

        Args:
            task_id: identifier of task to start.

        Returns:
            No data is returned.
        """
        if isinstance(task_id, int):
            task_id = str(task_id)
        try:
            task_to_start = self.not_started[task_id]
            self.working_on[task_id] = task_to_start
            del self.not_started[task_id]
            self.unsaved_changes = True
            print("Task #{0}: '{1}', started.".format(task_id,
                                                      task_to_start["task"]))
        except KeyError:
            print("There is no pending task with such id. Please check the" +
                  " id number of the task to mark as started again.")

    def completed_task(self, task_id):
        """Marks a current task as completed.

        Args:
            task_id: identifier of task to mark as completed.

        Returns:
            No data is returned.
        """
        if isinstance(task_id, int):
            task_id = str(task_id)
        try:
            completed_task = self.working_on[task_id]
            completed_task["completed"] = time.strftime("%Y/%m/%d")
            self.completed_tasks[task_id] = completed_task
            del self.working_on[task_id]
            self.unsaved_changes = True
            print("Task #{0}: '{1}', completed.".format(task_id,
                                                        completed_task["task"])
                  )
        except KeyError:
            print("There is no current task with such id. Please check the" +
                  " id number of the task to mark as complete again.")

    @property
    def refresh(self):
        """Saves pending changes in the pending.json file."""
        if self.unsaved_changes:
            file_with_changes = {
                "next_task_id": self.next_task_id,
                "completed": self.completed_tasks,
                "working_on": self.working_on,
                "not_started": self.not_started
            }
            with open(self.dir_path, 'w') as pending_tasks_file:
                pending_tasks_file.write(
                    json.dumps(file_with_changes,
                               sort_keys=True,
                               indent=4,
                               separators=(",", ": "))
                )
            self.unsaved_changes = False
            print("Changes applied to pending tasks file.")
        else:
            print("No changes have been made to the pending tasks file.")

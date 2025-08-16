"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import Template
from rich.console import Console

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'MINI_SWE_AGENT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.budget_tokens = 131768
        self.remaining_tokens = self.budget_tokens
        self.console = Console(width=120)

    def render_template(self, template: str, **kwargs) -> str:
        cs = asdict(self.config) | asdict(self.env.config) | asdict(self.model.config) | platform.uname()._asdict()
        return Template(template).render(**kwargs, **cs, **os.environ)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def run(self, task: str) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template, task=task))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        instance_id = getattr(self, "instance_id", "N/A")
        q = self.query()
        o = self.get_observation(q)
        self.console.log(
            f"{instance_id=}, step={self.model.n_calls}/{self.config.step_limit}:\nquery={q}\nobservation={o}",
            markup=False
        )
        return o

    def query(self) -> dict:
        """Query the model and return the response."""
        if (
            0 < self.config.step_limit <= self.model.n_calls 
            or 0 < self.config.cost_limit <= self.model.cost 
            # or self.remaining_tokens == 0
        ):
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message(
            "assistant",
            f'<think>\n{response["reasoning_content"].strip()}\n</think>' \
            + response["content"]
        )
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))

        remaining_steps = self.config.step_limit - self.model.n_calls
        self.remaining_tokens = max(0, self.budget_tokens - response["usage"]["total_tokens"])
        output.update({
            "budget_turns": remaining_steps,
            "budget_tokens": self.remaining_tokens,
        })

        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(
            r"<tool: bash>\n(.*?)\n</tool>",
            response["content"],
            re.DOTALL
        )
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""

        if "MINI_SWE_AGENT_FINAL_OUTPUT" in (text := output.get("output", "")):
            lines = text.lstrip().splitlines()
            if lines and lines[0].strip() == "MINI_SWE_AGENT_FINAL_OUTPUT":
                raise Submitted("\n".join(lines[1:]))

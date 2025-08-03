import asyncio
from dataclasses import dataclass, field
from typing import Any

from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command as RexCommand


@dataclass
class SwerexModalEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    deployment_extra_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs to pass to DockerDeployment."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""


class SwerexModalEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Docker container using SWE-ReX for sandboxing."""
        self.config = SwerexModalEnvironmentConfig(**kwargs)
        self.deployment = ModalDeployment(image=self.config.image, **self.config.deployment_extra_kwargs)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        output = asyncio.run(
            self.deployment.runtime.execute(
                RexCommand(
                    command=command,
                    shell=True,
                    check=False,
                    cwd=cwd or self.config.cwd,
                    timeout=self.config.timeout,
                    env=self.config.env
                )
            )
        )
        return {
            "output": f"<stdout>\n{output.stdout}</stdout>\n<stderr>\n{output.stderr}</stderr>",
            "returncode": output.exit_code,
        }

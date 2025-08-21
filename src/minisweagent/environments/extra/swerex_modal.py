import asyncio
import logging
import os
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

import modal
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command as RexCommand
from tenacity import Retrying, after_log, before_log, retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    forward_env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    deployment_retry: int = 20
    """Number of times to attempt to start a Modal container"""


class SwerexModalEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Docker container using SWE-ReX for sandboxing."""
        self.config = SwerexModalEnvironmentConfig(**kwargs)
        with modal.enable_output():
            self.deployment = ModalDeployment(
                image=self.config.image,
                **self.config.deployment_extra_kwargs
            )
            for attempt in Retrying(
                stop=stop_after_attempt(self.config.deployment_retry),
                wait=wait_exponential(max=60),
                before=before_log(logger, logging.INFO),
                after=after_log(logger, logging.INFO),
            ):
                with attempt:
                    asyncio.run(self.deployment.start())

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(max=60),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        env = self.config.env

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env[key] = value

        command = " ".join(f"{k}={v}" for k,v in env.items()) + " " + command
        command = f"bash -lc {shlex.quote(command)}"
        logger.info(command)

        output = asyncio.run(
            self.deployment.runtime.execute(
                RexCommand(
                    command=command,
                    shell=True,
                    check=False,
                    cwd=cwd or self.config.cwd,
                    timeout=self.config.timeout,
                    # the docker image already comes with the conda env in the PATH
                    # env var. setting the env argument overrides that and prevents
                    # the correct python binary from running.
                    # env=self.config.env
                )
            )
        )
        return {
            # "output": f"<stdout>\n{output.stdout}</stdout>\n<stderr>\n{output.stderr}</stderr>",
            "output": f"{output.stdout.strip()}{(f'\n\n' + output.stderr) if output.stderr else ''}",
            "returncode": output.exit_code,
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

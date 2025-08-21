import asyncio
import logging
import os
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

import modal
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command as RexCommand
from tenacity import Retrying, before_sleep_log, stop_after_attempt, wait_exponential

from minisweagent.utils.log import logger


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
    retry_attempts: int = 20
    """Number of times to attempt to start an unsafe function"""
    retry_max_wait: int = 60
    """Max number of seconds to wait before retries"""


class SwerexModalEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Modal Docker container using SWE-ReX for sandboxing."""
        self.config = SwerexModalEnvironmentConfig(**kwargs)
        self.retry = Retrying(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(max=self.config.retry_max_wait),
            before_sleep=before_sleep_log(logger, logging.INFO),
            reraise=True,
        )
        with modal.enable_output():
            self.deployment = ModalDeployment(
                image=self.config.image,
                **self.config.deployment_extra_kwargs
            )
            for attempt in self.retry:
                with attempt:
                    asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        env = self.config.env

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env[key] = value

        command = " ".join(f"{k}={v}" for k,v in env.items()) + " " + command
        command = f"bash -lc {shlex.quote(command)}"
        logger.info(command)

        for attempt in self.retry:
            with attempt:
                output = asyncio.run(
                    self.deployment.runtime.execute(
                        RexCommand(
                            command=command,
                            shell=True,
                            check=False,
                            cwd=cwd or self.config.cwd,
                            timeout=self.config.timeout,
                            merge_output_streams=True,
                        )
                    )
                )
                return {
                    "output": output.stdout,
                    "returncode": output.exit_code,
                }

        # unreachable, but we appease the type-checker
        raise ValueError("Retry loop exited without returning")

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

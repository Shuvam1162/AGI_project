"""
nars_runner.py
--------------
Subprocess-based runner for OpenNARS 3.1.x CLI.

Usage:
    runner = NARSRunner(jar_path="OpenNARS-3.1.0.jar")
    f, c = runner.run(task_file="task1.nal", max_cycles=500, concept_bag_size=500)

The runner:
  1. Writes a temporary Narsese script that includes the task file and
     appends the required cycle count command.
  2. Launches OpenNARS as a subprocess with the given parameters.
  3. Parses stdout to extract the truth value for the queried statement.
  4. Returns (frequency, confidence).

If OpenNARS is not available, raises FileNotFoundError — in that case use
NARSSimulator from nars_simulator.py instead.
"""

import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Tuple, Optional

from harness.output_parser import extract_best_answer


class NARSRunner:
    """
    Runs OpenNARS experiments via CLI subprocess.

    Parameters
    ----------
    jar_path : str
        Path to the OpenNARS JAR file (e.g., OpenNARS-3.1.0.jar).
    java_cmd : str
        Java executable (default: "java").
    timeout : int
        Maximum seconds to wait for a single run (default: 120).
    """

    def __init__(
        self,
        jar_path: str = "OpenNARS-3.1.0.jar",
        java_cmd: str = "java",
        timeout: int = 120,
    ):
        self.jar_path = Path(jar_path)
        self.java_cmd = java_cmd
        self.timeout = timeout

        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"OpenNARS JAR not found: {self.jar_path}. "
                "Download from https://github.com/opennars/opennars/releases "
                "or use NARSSimulator for synthetic experiments."
            )

    def run(
        self,
        task_file: str,
        max_cycles: int,
        concept_bag_size: int,
        seed: int = 0,
    ) -> Tuple[float, float]:
        """
        Execute a single NARS run.

        Parameters
        ----------
        task_file : str
            Path to the Narsese (.nal) input file.
        max_cycles : int
            Maximum inference cycles (Nc).
        concept_bag_size : int
            Concept memory capacity (Nm).
        seed : int
            Random seed (controls stochastic priority sampling).

        Returns
        -------
        (frequency, confidence) : Tuple[float, float]
            Extracted truth value from the best answer line.
        """
        task_path = Path(task_file)
        if not task_path.exists():
            raise FileNotFoundError(f"Task file not found: {task_path}")

        # Build the Narsese script with cycle control appended
        with open(task_path) as f:
            task_content = f.read()

        script = (
            f"{task_content}\n"
            f"*setCyclesBetweenGC=1\n"
            f"*volume=100\n"
            f"*cycles={max_cycles}\n"
            f"*stop\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nal", delete=False
        ) as tmp:
            tmp.write(script)
            tmp_path = tmp.name

        try:
            cmd = [
                self.java_cmd,
                f"-DconceptBagSize={concept_bag_size}",
                f"-DrandomSeed={seed}",
                "-jar", str(self.jar_path),
                tmp_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            f_val, c_val = extract_best_answer(output)
            return f_val, c_val

        except subprocess.TimeoutExpired:
            # Treat timeout as failed run — return null belief
            return 0.5, 0.0
        finally:
            os.unlink(tmp_path)

    def is_available(self) -> bool:
        """Check whether OpenNARS JAR is accessible."""
        return self.jar_path.exists()

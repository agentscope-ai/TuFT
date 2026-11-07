"""Run the existing coding benchmark using the host's isolated tool service."""

import benchmark_shared_weight_agents
import coding_tasks
from server_coding_tools import ServerTester


if __name__ == "__main__":
    coding_tasks.SandboxTester = ServerTester
    benchmark_shared_weight_agents.main()

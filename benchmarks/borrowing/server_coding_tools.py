"""Isolated Docker tools for the SSH benchmark; never executes candidates on the host."""

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from coding_tasks import HARNESS


class ServerTester:
    def __init__(self):
        self.records = []

    async def start(self):
        self.socket = os.environ["TUFT_CODING_TOOL_SOCKET"]

    async def evaluate(self, code, test_cases, *, constants=None, **identity):
        start = time.perf_counter()
        reader, writer = await asyncio.open_unix_connection(self.socket, limit=2**20)
        try:
            payload = dict(code=code, cases=test_cases, constants=constants or {})
            writer.write(json.dumps(payload).encode() + b"\n")
            await writer.drain()
            reply = json.loads(await asyncio.wait_for(reader.readline(), 120))
            if "error" in reply:
                raise RuntimeError(reply["error"])
            self.records.append(dict(**identity, **reply))
            return reply["result"], time.perf_counter() - start
        finally:
            writer.close()
            await writer.wait_closed()


async def serve(socket, image, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async def handle(reader, writer):
        name = "tuft-tool-" + uuid.uuid4().hex
        process = None
        try:
            payload = json.loads(await asyncio.wait_for(reader.readline(), 10))
            assert isinstance(payload["code"], str) and isinstance(payload["cases"], list)
            queued = time.perf_counter()
            async with semaphore:
                begin = time.perf_counter()
                process = await asyncio.create_subprocess_exec(
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    name,
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=64",
                    "--memory=512m",
                    "--cpus=1",
                    "--user=65534:65534",
                    "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
                    "--entrypoint=python",
                    image,
                    "-I",
                    "-c",
                    HARNESS,
                    json.dumps(payload),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    out, err = await asyncio.wait_for(process.communicate(), 15)
                    try:
                        result = json.loads(out) if process.returncode == 0 else None
                    except ValueError:
                        result = None
                    if not isinstance(result, dict) or "passed" not in result:
                        result = dict(
                            passed=0,
                            total=len(payload["cases"]),
                            failures=[dict(error=f"exit={process.returncode}: {err[-500:]!r}")],
                        )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    result = dict(
                        passed=0,
                        total=len(payload["cases"]),
                        failures=[dict(error="Tool execution timed out")],
                    )
                finally:
                    cleanup = await asyncio.create_subprocess_exec(
                        "docker",
                        "rm",
                        "-f",
                        name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await cleanup.wait()
                response = dict(
                    result=result,
                    queue_s=begin - queued,
                    execution_s=time.perf_counter() - begin,
                    sandbox_id=name,
                )
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        except Exception as exc:
            writer.write(json.dumps(dict(error=str(exc))).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_unix_server(handle, path=socket, limit=2**20)
    os.chmod(socket, 0o600)
    print("TOOL_SERVER_READY", socket, flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if Path(args.socket).exists():
        raise SystemExit("Refusing to replace existing socket")
    asyncio.run(serve(args.socket, args.image, args.concurrency))

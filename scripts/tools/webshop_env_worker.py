#!/usr/bin/env python3
"""JSON-lines bridge to the pinned WebShop text environment."""

from __future__ import annotations

import contextlib
import json
import sys

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def main() -> None:
    with contextlib.redirect_stdout(sys.stderr):
        env = WebAgentTextEnv(observation_mode="text", num_products=1000)
    emit({"ready": True})
    for line in sys.stdin:
        request = json.loads(line)
        command = request["command"]
        try:
            if command == "reset":
                with contextlib.redirect_stdout(sys.stderr):
                    env.reset(session=int(request["session"]))
                emit(
                    {
                        "observation": env.observation,
                        "actions": env.get_available_actions(),
                    }
                )
            elif command == "step":
                observation, reward, done, _ = env.step(request["action"])
                emit(
                    {
                        "observation": observation,
                        "reward": reward,
                        "done": done,
                        "actions": env.get_available_actions() if not done else {},
                    }
                )
            elif command == "close":
                env.close()
                emit({"closed": True})
                return
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            emit({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()

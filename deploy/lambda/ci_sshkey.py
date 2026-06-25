#!/usr/bin/env python
"""CI helper: register or delete an ephemeral Lambda Cloud SSH key.

So the deploy-e2e workflow can SSH into a freshly-launched instance with only LAMBDA_API_KEY
(no pre-shared key): generate a throwaway keypair, register the public half here, use it to
launch + verify, then delete it. Reuses LambdaClient from launch.py.

    python ci_sshkey.py register <name> <public-key-file>   # prints the new key's id
    python ci_sshkey.py delete   <key-id>
"""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch import LambdaClient  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        sys.exit("LAMBDA_API_KEY not set")
    client = LambdaClient(api_key)
    cmd = sys.argv[1]
    if cmd == "register":
        name, pubfile = sys.argv[2], sys.argv[3]
        pub = open(pubfile).read().strip()
        data = client.request("POST", "/ssh-keys", {"name": name, "public_key": pub}).get(
            "data", {}
        )
        print(data.get("id", ""))
    elif cmd == "delete":
        client.request("DELETE", "/ssh-keys/%s" % sys.argv[2])
        print("deleted %s" % sys.argv[2])
    else:
        sys.exit("unknown command: %s (use register|delete)" % cmd)


if __name__ == "__main__":
    main()

# WebShop benchmark

The environment is the official Princeton WebShop code pinned at commit
`64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`. This run uses its official
1,000-product small setting, mirrored by `zhangdw/webshop` revision
`2fa2bd5dc0cf227e98f6512e71fb88f59eb0c741`, not the full catalog.

Prepare it with `sudo apt-get install -y openjdk-17-jdk` followed by
`scripts/setup/setup_webshop.sh`.

The chatbot emits one action and the environment evaluates it. The agent may
call `webshop_action` up to six times. The tool accepts `search[keywords]` or
`click[visible text]`; a child process executes it in the official text
environment, and that child's CPU is included. Sessions 0--4 are measured
after one warm-up. Reward is the official environment reward.

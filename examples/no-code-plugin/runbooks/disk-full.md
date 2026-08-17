# Disk full

1. It is almost always logs: check `/var/log` first.
2. Truncate, never delete — a deleted open file frees nothing:
   `truncate -s 0 <file>`.
3. If logs were not it, find the real growth: `du -x --max-depth=2 / | sort -h`.
4. Growth in the database volume is an escalation, not a cleanup.

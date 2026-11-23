#!/bin/sh
set -eu
umask 077
python check_cpu.py runtime
nice -n 15 ionice -c 3 python convert.py
exec nice -n 15 ionice -c 3 python verify.py

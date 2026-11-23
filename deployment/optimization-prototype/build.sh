#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Compiler scratch must remain inside the authorized directory.
export TMPDIR="$PWD/build-tmp"
export LC_ALL=C
# No -march=native, no F16C, no fast-math/reassociation, no extra worker threads.
gcc -pipe -std=c11 -O2 -fPIC -shared -mno-f16c -fno-fast-math -ffp-contract=off -fexcess-precision=standard -Wall -Wextra -Werror convert.c verify.c -lm -o libprototype.so
gcc -pipe -std=c11 -O1 -g -fPIC -shared -mno-f16c -fno-fast-math -ffp-contract=off -fexcess-precision=standard -fsanitize=undefined -fno-sanitize-recover=all -Wall -Wextra -Werror convert.c verify.c -lm -o libprototype-ubsan.so

#!/usr/bin/env bash
# Install all system + Python dependencies needed to run this project.
#
# Tested on Ubuntu 24.04 (noble). Requires apt + sudo.
# Usage: ./scripts/setup.sh [--verify]
#   --verify   run the smoke test after install to confirm everything works
set -euo pipefail

VERIFY=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY=1 ;;
        -h|--help) sed -n '2,7p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Resolve project root (this script lives in <project>/scripts/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
c_green=$'\033[1;32m'; c_reset=$'\033[0m'
log()  { printf '%s[setup]%s %s\n' "$c_blue" "$c_reset" "$*"; }
warn() { printf '%s[warn] %s %s\n' "$c_yellow" "$c_reset" "$*"; }
err()  { printf '%s[err]  %s %s\n' "$c_red" "$c_reset" "$*" >&2; }

# ---------------------------------------------------------------------------
# 0. Pre-flight
# ---------------------------------------------------------------------------
if ! command -v apt-get >/dev/null; then
    err "this script targets apt-based distros (Ubuntu/Debian)."
    err "for other systems, install manually: Python 3.11, OpenJDK 17, then run pip install -r requirements.txt && pip install -e ."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Python 3.11
# ---------------------------------------------------------------------------
# 3.11 is required: Spark 3.5.1 imports `distutils` which was removed in 3.12,
# and PySpark workers must match the driver's minor version.
if command -v python3.11 >/dev/null; then
    log "Python 3.11 already installed: $(python3.11 --version)"
else
    log "installing Python 3.11 (deadsnakes PPA)"
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# ---------------------------------------------------------------------------
# 2. OpenJDK 17
# ---------------------------------------------------------------------------
# Spark 3.5 officially supports Java 8 / 11 / 17. Java 21+ sealed the
# sun.misc.Unsafe and DirectByteBuffer paths Arrow needs; --add-opens does
# not restore them, so Arrow-based pandas→Spark conversion fails.
find_java17() {
    local c
    for c in /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/java-17-openjdk* \
             /usr/lib/jvm/temurin-17-jdk-* /usr/lib/jvm/zulu17-* ; do
        [[ -d "$c" ]] && { echo "$c"; return 0; }
    done
    return 1
}

if JAVA_17_HOME="$(find_java17)"; then
    log "OpenJDK 17 already installed: $JAVA_17_HOME"
else
    log "installing OpenJDK 17"
    sudo apt-get install -y openjdk-17-jdk
    JAVA_17_HOME="$(find_java17)" || { err "openjdk-17-jdk install reported success but JVM not found"; exit 1; }
fi

# ---------------------------------------------------------------------------
# 3. Virtualenv
# ---------------------------------------------------------------------------
VENV_DIR="$PROJECT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
    venv_py="$("$VENV_DIR/bin/python" --version 2>&1 || true)"
    if [[ "$venv_py" == *"3.11"* ]]; then
        log "virtualenv at $VENV_DIR already on Python 3.11 — reusing"
    else
        warn "virtualenv at $VENV_DIR is $venv_py — recreating with Python 3.11"
        rm -rf "$VENV_DIR"
        python3.11 -m venv "$VENV_DIR"
    fi
else
    log "creating virtualenv at $VENV_DIR"
    python3.11 -m venv "$VENV_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Python dependencies
# ---------------------------------------------------------------------------
log "installing Python dependencies"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
printf '\n%s[ok]%s setup complete.\n\n' "$c_green" "$c_reset"
cat <<EOF
  Python  : $("$VENV_DIR/bin/python" --version)
  Java 17 : $JAVA_17_HOME
  Venv    : $VENV_DIR

Next:
  source $VENV_DIR/bin/activate
  python scripts/smoke_test_io.py

The smoke test pins JAVA_HOME, PYSPARK_PYTHON, and the JVM --add-opens
flags internally — no extra shell setup required.
EOF

# ---------------------------------------------------------------------------
# 6. Optional verification
# ---------------------------------------------------------------------------
if [[ "$VERIFY" == "1" ]]; then
    log "running smoke test"
    "$VENV_DIR/bin/python" "$PROJECT_DIR/scripts/smoke_test_io.py"
fi

#!/usr/bin/env bash
# Install all system and Python dependencies needed to run this project.
#
# Supported system installers:
#   - Ubuntu/Debian: apt-get
#   - Fedora/RHEL: dnf or yum
#   - macOS: Homebrew
#
# Usage: ./scripts/setup.sh [--verify] [--skip-system]
#   --verify       run smoke tests after install
#   --skip-system  do not install Python or Java; require them to exist already
set -euo pipefail

VERIFY=0
SKIP_SYSTEM=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY=1 ;;
        --skip-system) SKIP_SYSTEM=1 ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Resolve project root (this script lives in <project>/scripts/).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
c_green=$'\033[1;32m'; c_reset=$'\033[0m'
log()  { printf '%s[setup]%s %s\n' "$c_blue" "$c_reset" "$*"; }
warn() { printf '%s[warn] %s %s\n' "$c_yellow" "$c_reset" "$*"; }
err()  { printf '%s[err]  %s %s\n' "$c_red" "$c_reset" "$*" >&2; }

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

run_as_root() {
    if [[ "$(id -u)" == "0" ]]; then
        "$@"
    elif has_cmd sudo; then
        sudo "$@"
    else
        err "sudo is required to install system packages. Install Python 3.11 and Java 17 manually, then rerun with --skip-system."
        exit 1
    fi
}

resolve_exe() {
    local candidate="$1"
    if [[ "$candidate" == */* ]]; then
        [[ -x "$candidate" ]] && echo "$candidate"
    else
        command -v "$candidate" 2>/dev/null || true
    fi
}

is_python311() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

find_python311() {
    local candidate resolved
    for candidate in "${PYTHON_BIN:-}" python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
        [[ -n "$candidate" ]] || continue
        resolved="$(resolve_exe "$candidate" || true)"
        [[ -n "$resolved" ]] || continue
        if is_python311 "$resolved"; then
            echo "$resolved"
            return 0
        fi
    done
    return 1
}

install_python311() {
    if [[ "$SKIP_SYSTEM" == "1" ]]; then
        err "Python 3.11 was not found. Install it manually or rerun without --skip-system."
        exit 1
    fi

    if has_cmd apt-get; then
        log "installing Python 3.11 with apt-get"
        run_as_root apt-get update -y
        if ! run_as_root apt-get install -y python3.11 python3.11-venv python3.11-dev; then
            log "apt did not provide Python 3.11 directly; trying deadsnakes PPA"
            run_as_root apt-get install -y software-properties-common
            run_as_root add-apt-repository -y ppa:deadsnakes/ppa
            run_as_root apt-get update -y
            run_as_root apt-get install -y python3.11 python3.11-venv python3.11-dev
        fi
    elif has_cmd dnf; then
        log "installing Python 3.11 with dnf"
        run_as_root dnf install -y python3.11 python3.11-devel python3.11-pip
    elif has_cmd yum; then
        log "installing Python 3.11 with yum"
        run_as_root yum install -y python3.11 python3.11-devel python3.11-pip
    elif has_cmd brew; then
        log "installing Python 3.11 with Homebrew"
        brew install python@3.11
    else
        err "could not find a supported package manager for Python 3.11."
        err "Install Python 3.11 manually, then rerun: ./scripts/setup.sh --skip-system"
        exit 1
    fi
}

is_java17_home() {
    local java_home="$1"
    [[ -x "$java_home/bin/java" ]] || return 1
    "$java_home/bin/java" -version 2>&1 | grep -Eq 'version "17\.'
}

find_java17() {
    local candidate java_bin java_home

    if [[ -n "${JAVA_HOME:-}" ]] && is_java17_home "$JAVA_HOME"; then
        echo "$JAVA_HOME"
        return 0
    fi

    if [[ -x /usr/libexec/java_home ]]; then
        java_home="$(/usr/libexec/java_home -v 17 2>/dev/null || true)"
        if [[ -n "$java_home" ]] && is_java17_home "$java_home"; then
            echo "$java_home"
            return 0
        fi
    fi

    for candidate in \
        /usr/lib/jvm/java-17-openjdk-amd64 \
        /usr/lib/jvm/java-17-openjdk \
        /usr/lib/jvm/java-17-openjdk-* \
        /usr/lib/jvm/temurin-17-jdk-* \
        /usr/lib/jvm/zulu17-* \
        /opt/homebrew/opt/openjdk@17 \
        /usr/local/opt/openjdk@17; do
        if [[ -d "$candidate" ]] && is_java17_home "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    if has_cmd brew; then
        candidate="$(brew --prefix openjdk@17 2>/dev/null || true)"
        if [[ -n "$candidate" ]] && is_java17_home "$candidate"; then
            echo "$candidate"
            return 0
        fi
    fi

    java_bin="$(command -v java 2>/dev/null || true)"
    if [[ -n "$java_bin" ]]; then
        java_home="$(dirname "$(dirname "$(readlink -f "$java_bin" 2>/dev/null || echo "$java_bin")")")"
        if is_java17_home "$java_home"; then
            echo "$java_home"
            return 0
        fi
    fi

    return 1
}

install_java17() {
    if [[ "$SKIP_SYSTEM" == "1" ]]; then
        err "Java 17 was not found. Install it manually or rerun without --skip-system."
        exit 1
    fi

    if has_cmd apt-get; then
        log "installing OpenJDK 17 with apt-get"
        run_as_root apt-get update -y
        run_as_root apt-get install -y openjdk-17-jdk
    elif has_cmd dnf; then
        log "installing OpenJDK 17 with dnf"
        run_as_root dnf install -y java-17-openjdk-devel
    elif has_cmd yum; then
        log "installing OpenJDK 17 with yum"
        run_as_root yum install -y java-17-openjdk-devel
    elif has_cmd brew; then
        log "installing OpenJDK 17 with Homebrew"
        brew install openjdk@17
    else
        err "could not find a supported package manager for Java 17."
        err "Install Java 17 manually, then rerun: ./scripts/setup.sh --skip-system"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
# Python 3.11 is required because Spark 3.5.1 is not happy under Python 3.12
# in this project, and PySpark workers must match the driver minor version.
if PYTHON_311="$(find_python311)"; then
    log "Python 3.11 already installed: $("$PYTHON_311" --version)"
else
    install_python311
    PYTHON_311="$(find_python311)" || { err "Python 3.11 install completed but python3.11 was not found"; exit 1; }
fi

# Spark 3.5 supports Java 8, 11, and 17. This project standardizes on Java 17.
if JAVA_17_HOME="$(find_java17)"; then
    log "Java 17 already installed: $JAVA_17_HOME"
else
    install_java17
    JAVA_17_HOME="$(find_java17)" || { err "Java 17 install completed but JAVA_HOME could not be resolved"; exit 1; }
fi

export JAVA_HOME="$JAVA_17_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

# ---------------------------------------------------------------------------
# 2. Virtualenv
# ---------------------------------------------------------------------------
VENV_DIR="$PROJECT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
    venv_py="$("$VENV_DIR/bin/python" --version 2>&1 || true)"
    if [[ "$venv_py" == *"3.11"* ]]; then
        log "virtualenv at $VENV_DIR already uses Python 3.11; reusing"
    else
        warn "virtualenv at $VENV_DIR is $venv_py; recreating with Python 3.11"
        rm -rf "$VENV_DIR"
        "$PYTHON_311" -m venv "$VENV_DIR"
    fi
else
    log "creating virtualenv at $VENV_DIR"
    "$PYTHON_311" -m venv "$VENV_DIR"
fi

VENV_BIN="$VENV_DIR/bin"
export PYSPARK_PYTHON="$VENV_BIN/python"
export PYSPARK_DRIVER_PYTHON="$VENV_BIN/python"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
log "installing Python dependencies from requirements.txt"
"$VENV_BIN/python" -m pip install --upgrade pip setuptools wheel
"$VENV_BIN/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

log "installing project in editable mode"
"$VENV_BIN/python" -m pip install -e "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 4. Dependency verification
# ---------------------------------------------------------------------------
log "verifying Python dependency graph"
"$VENV_BIN/python" -m pip check

log "verifying local HD-BET executable"
"$VENV_BIN/hd-bet" --help >/dev/null

log "verifying project CLI"
"$VENV_BIN/python" -m wmh_spark.preprocessing.skull_strip --help >/dev/null

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
printf '\n%s[ok]%s setup complete.\n\n' "$c_green" "$c_reset"
cat <<EOF
  Python  : $("$VENV_BIN/python" --version)
  Java 17 : $JAVA_HOME
  Venv    : $VENV_DIR
  HD-BET  : $VENV_BIN/hd-bet

Next:
  source $VENV_DIR/bin/activate
  python scripts/smoke_test_io.py

Run skull stripping:
  python -m wmh_spark.preprocessing.skull_strip --input-dir ../datasets --output-dir data/work/skull_stripped --smoke-test --device cpu
EOF

# ---------------------------------------------------------------------------
# 6. Optional verification
# ---------------------------------------------------------------------------
if [[ "$VERIFY" == "1" ]]; then
    log "running smoke test"
    "$VENV_BIN/python" "$PROJECT_DIR/scripts/smoke_test_io.py"
fi

#!/usr/bin/env bash
# Compile and run a COMSOL Java API class (Linux example).
# Usage: COMSOL_ROOT=/path/to/comsol64 bash tools/jrun.sh ChainedRunner2 <model.mph> <jobs.csv>
set -euo pipefail
: "${COMSOL_ROOT:?Set COMSOL_ROOT to the COMSOL installation directory}"
CROOT="$COMSOL_ROOT"
JAVA="$CROOT/java/glnxa64/jre/bin"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../comsol/java" && pwd)"
CLASSPATH="$(find "$CROOT/plugins" -maxdepth 1 -type f -name '*.jar' -printf '%p:' | tr -d '
')"
export LD_LIBRARY_PATH="$CROOT/lib/glnxa64:$CROOT/plugins/glnxa64:${LD_LIBRARY_PATH:-}"
CLASS="${1:?ClassName required}"; shift
cd "$DIR"
"$JAVA/javac" -proc:none -cp "$CLASSPATH" "${CLASS}.java"
exec "$JAVA/java" -cp "${CLASSPATH}${DIR}" -Djava.library.path="$CROOT/bin/glnxa64" "$CLASS" "$@"

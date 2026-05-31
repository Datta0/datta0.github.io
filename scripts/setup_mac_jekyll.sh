#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_mac_jekyll.sh [options]

Options:
  --repo URL          Clone/update this repo, then run setup from that checkout.
  --branch BRANCH     Branch to use with --repo. Default: rl_for_llm.
  --dir DIR           Checkout directory for --repo. Default: ~/datta0.github.io.
  --no-serve          Install dependencies but do not start Jekyll.
  --no-open           Do not open the local site in the browser.
  --port PORT         Port for local server. Default: first free port from 4000.
  --livereload-port PORT
                     Port for LiveReload. Default: first free port from 35729.
  --host HOST         Host for local server. Default: 127.0.0.1.
  --install-homebrew  Install Homebrew/Linuxbrew if Ruby is missing or too old.
  -h, --help          Show this help.

What it does:
  1. Detects macOS/Linux and installs the basic local dependencies it needs.
  2. Optionally clones/updates the requested branch.
  3. Uses a valid system Ruby or Homebrew/Linuxbrew ruby@3.4 when needed.
  4. Installs bundler and gems into a Ruby-specific local bundle path.
  5. Runs `bundle exec jekyll serve`.
EOF
}

clone_repo=""
clone_branch="rl_for_llm"
clone_dir=""
serve=1
open_browser=1
port=""
livereload_port=""
host="127.0.0.1"
install_homebrew=0
forwarded_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      clone_repo="${2:?Missing value for --repo}"
      shift 2
      ;;
    --branch)
      clone_branch="${2:?Missing value for --branch}"
      shift 2
      ;;
    --dir)
      clone_dir="${2:?Missing value for --dir}"
      shift 2
      ;;
    --no-serve)
      serve=0
      forwarded_args+=("$1")
      shift
      ;;
    --no-open)
      open_browser=0
      forwarded_args+=("$1")
      shift
      ;;
    --port)
      port="${2:?Missing value for --port}"
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    --livereload-port)
      livereload_port="${2:?Missing value for --livereload-port}"
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    --host)
      host="${2:?Missing value for --host}"
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    --install-homebrew)
      install_homebrew=1
      forwarded_args+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

os_name="$(uname -s)"
if [[ "$os_name" != "Darwin" && "$os_name" != "Linux" ]]; then
  echo "This script supports macOS and Linux. Detected: $os_name" >&2
  exit 1
fi

sudo_cmd=()
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  sudo_cmd=(sudo)
fi

install_linux_packages() {
  if [[ "$os_name" != "Linux" ]]; then
    return 0
  fi

  if [[ "${#sudo_cmd[@]}" -gt 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Linux packages. Install dependencies manually or rerun as root." >&2
    return 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y build-essential procps curl file git ruby-full ruby-dev zlib1g-dev libffi-dev libyaml-dev
  elif command -v dnf >/dev/null 2>&1; then
    "${sudo_cmd[@]}" dnf install -y gcc gcc-c++ make procps-ng curl file git ruby ruby-devel zlib-devel libffi-devel libyaml-devel openssl-devel
  elif command -v yum >/dev/null 2>&1; then
    "${sudo_cmd[@]}" yum install -y gcc gcc-c++ make procps-ng curl file git ruby ruby-devel zlib-devel libffi-devel libyaml-devel openssl-devel
  elif command -v pacman >/dev/null 2>&1; then
    "${sudo_cmd[@]}" pacman -Sy --needed --noconfirm base-devel procps-ng curl file git ruby zlib libffi libyaml openssl
  elif command -v zypper >/dev/null 2>&1; then
    "${sudo_cmd[@]}" zypper install -y gcc gcc-c++ make procps curl file git ruby ruby-devel zlib-devel libffi-devel libyaml-devel libopenssl-devel
  else
    cat >&2 <<'EOF'
Could not detect a supported Linux package manager.
Please install these first: git, curl, file, procps, gcc/make, Ruby >= 3.1, ruby-dev.
EOF
    return 1
  fi
}

if [[ "$os_name" == "Darwin" ]]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "macOS command line tools are missing."
    echo "A dialog will open. Install them, then rerun this script."
    xcode-select --install || true
    exit 1
  fi
elif [[ "$os_name" == "Linux" ]]; then
  missing_basic=0
  for cmd in git curl file make gcc; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing_basic=1
    fi
  done
  if [[ "$missing_basic" -eq 1 ]]; then
    echo "Installing Linux basics..."
    install_linux_packages
  fi
fi

if [[ -n "$clone_repo" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required. Install command line tools, then rerun this script." >&2
    exit 1
  fi

  if [[ -z "$clone_dir" ]]; then
    clone_dir="$HOME/datta0.github.io"
  fi

  if [[ -d "$clone_dir/.git" ]]; then
    echo "Updating existing checkout: $clone_dir"
    git -C "$clone_dir" fetch origin "$clone_branch"
    if git -C "$clone_dir" show-ref --verify --quiet "refs/heads/$clone_branch"; then
      git -C "$clone_dir" checkout "$clone_branch"
    else
      git -C "$clone_dir" checkout -b "$clone_branch" "origin/$clone_branch"
    fi
    git -C "$clone_dir" pull --ff-only origin "$clone_branch"
  elif [[ -e "$clone_dir" ]]; then
    echo "$clone_dir exists but is not a git checkout. Choose another --dir." >&2
    exit 1
  else
    echo "Cloning $clone_repo branch $clone_branch into $clone_dir"
    mkdir -p "$(dirname "$clone_dir")"
    git clone --branch "$clone_branch" --single-branch "$clone_repo" "$clone_dir"
  fi

  exec bash "$clone_dir/scripts/setup_mac_jekyll.sh" "${forwarded_args[@]}"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

if [[ ! -f Gemfile ]]; then
  echo "Could not find Gemfile. Run this script from the datta0.github.io repo." >&2
  exit 1
fi

echo "Repo: $repo_root"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
fi

ensure_brew_path() {
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  elif [[ -x /home/linuxbrew/.linuxbrew/bin/brew ]]; then
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
  elif [[ -x "$HOME/.linuxbrew/bin/brew" ]]; then
    eval "$("$HOME/.linuxbrew/bin/brew" shellenv)"
  fi
}

ensure_brew_path

ruby_formula="ruby@3.4"
if [[ -f .ruby-version ]]; then
  ruby_version="$(tr -d '[:space:]' < .ruby-version)"
  ruby_formula="ruby@${ruby_version%.*}"
fi

ruby_version_ok() {
  command -v ruby >/dev/null 2>&1 &&
    ruby -e 'v = Gem::Version.new(RUBY_VERSION); exit(v >= Gem::Version.new("3.1") && v < Gem::Version.new("4.0") ? 0 : 1)' >/dev/null 2>&1
}

ruby_ok=0
if ruby_version_ok; then
  ruby_ok=1
elif [[ "$os_name" == "Linux" ]]; then
  echo "Trying Linux package manager Ruby first..."
  install_linux_packages
  if ruby_version_ok; then
    ruby_ok=1
  fi
fi

if [[ "$ruby_ok" -ne 1 ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    if [[ "$install_homebrew" -eq 1 ]]; then
      if [[ "$os_name" == "Linux" ]]; then
        echo "Installing Linux prerequisites for Linuxbrew..."
        install_linux_packages
      fi
      echo "Installing Homebrew/Linuxbrew..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      ensure_brew_path
    else
      cat >&2 <<EOF
Ruby >= 3.1 and < 4.0 was not found.

This script can install $ruby_formula via Homebrew/Linuxbrew:
  bash scripts/setup_mac_jekyll.sh --install-homebrew

If you are running from GitHub:
  curl -fsSL <script-url> | bash -s -- --repo <repo-url> --branch <branch> --install-homebrew
EOF
      exit 1
    fi
  fi

  echo "Installing $ruby_formula via Homebrew/Linuxbrew..."
  brew install "$ruby_formula"
fi

if command -v brew >/dev/null 2>&1 && brew --prefix "$ruby_formula" >/dev/null 2>&1; then
  ruby_prefix="$(brew --prefix "$ruby_formula")"
  export PATH="$ruby_prefix/bin:$PATH"
fi

if [[ "$os_name" == "Linux" ]]; then
  if ! ruby -rrbconfig -e 'exit File.directory?(RbConfig::CONFIG["rubyhdrdir"]) ? 0 : 1' >/dev/null 2>&1; then
    echo "Ruby development headers are missing. Installing Linux Ruby build dependencies..."
    install_linux_packages
  fi
fi

gem_bindir="$(ruby -rrubygems -e 'print Gem.bindir')"
export PATH="$gem_bindir:$PATH"

echo "Ruby: $(ruby -v)"
echo "Gem:  $(gem -v)"

bundler_version="2.3.26"
if [[ -f Gemfile.lock ]]; then
  locked_bundler_version="$(awk '/^BUNDLED WITH$/ { getline; gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print; exit }' Gemfile.lock)"
  if [[ -n "$locked_bundler_version" ]]; then
    bundler_version="$locked_bundler_version"
  fi
fi

if ! gem list -i bundler -v "$bundler_version" >/dev/null 2>&1; then
  echo "Installing bundler $bundler_version..."
  gem install bundler -v "$bundler_version" --no-document
fi
bundle_cmd=(bundle "_${bundler_version}_")

echo "Bundler: $("${bundle_cmd[@]}" -v)"

ruby_source="custom"
if ruby -rrbconfig -e 'exit RbConfig::CONFIG["prefix"].include?("homebrew") ? 0 : 1' >/dev/null 2>&1; then
  ruby_source="homebrew"
fi
ruby_id="$(ruby -rrbconfig -e 'print "ruby-#{RUBY_VERSION}-#{RbConfig::CONFIG["host"]}"')-$ruby_source"
bundle_path=".bundle-gems/$ruby_id"

echo "Bundle path: $bundle_path"
"${bundle_cmd[@]}" config set path "$bundle_path"
"${bundle_cmd[@]}" install

if [[ "$serve" -eq 0 ]]; then
  echo "Setup complete."
  exit 0
fi

port_is_free() {
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ! ss -ltn "sport = :$1" | grep -q LISTEN
  else
    ruby -rsocket -e 's = TCPServer.new("127.0.0.1", ARGV[0].to_i); s.close' "$1" >/dev/null 2>&1
  fi
}

if [[ -z "$port" ]]; then
  for candidate in $(seq 4000 4010); do
    if port_is_free "$candidate"; then
      port="$candidate"
      break
    fi
  done

  if [[ -z "$port" ]]; then
    echo "Could not find a free port from 4000 to 4010. Pass --port PORT." >&2
    exit 1
  fi
elif ! port_is_free "$port"; then
  echo "Port $port is already in use. Pick another one with --port PORT." >&2
  exit 1
fi

if [[ -z "$livereload_port" ]]; then
  for candidate in $(seq 35729 35739); do
    if port_is_free "$candidate"; then
      livereload_port="$candidate"
      break
    fi
  done

  if [[ -z "$livereload_port" ]]; then
    echo "Could not find a free LiveReload port from 35729 to 35739. Pass --livereload-port PORT." >&2
    exit 1
  fi
elif ! port_is_free "$livereload_port"; then
  echo "LiveReload port $livereload_port is already in use. Pick another one with --livereload-port PORT." >&2
  exit 1
fi

open_host="$host"
if [[ "$open_host" == "0.0.0.0" ]]; then
  open_host="127.0.0.1"
fi
url="http://$open_host:$port/posts/systems-for-llm-rl/"

echo
echo "Starting Jekyll..."
echo "Open: $url"
echo "LiveReload port: $livereload_port"

if [[ "$open_browser" -eq 1 ]]; then
  (
    for _ in $(seq 1 60); do
      if curl -fsS "$url" >/dev/null 2>&1; then
        if [[ "$os_name" == "Darwin" ]]; then
          open "$url" >/dev/null 2>&1 || true
        elif command -v xdg-open >/dev/null 2>&1; then
          xdg-open "$url" >/dev/null 2>&1 || true
        fi
        exit 0
      fi
      sleep 1
    done
  ) &
fi

"${bundle_cmd[@]}" exec jekyll serve --livereload --livereload-port "$livereload_port" --host "$host" --port "$port"

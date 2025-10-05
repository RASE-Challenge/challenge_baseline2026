import os
import argparse
import yaml
import zipfile
import subprocess
import datetime 

OUTPUT_DIR = "/src/logs"

def run_cmd(cmd):
    """Run a shell command and return its output as text."""
    return subprocess.check_output(cmd, shell=True, text=True)

# Collect files/folders to zip
def save_pip_packages(timestamp, output_dir=f"{OUTPUT_DIR}/pip_freeze_logs"):
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamped filename
    filename = f"requirements_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    # Run pip freeze and capture output
    with open(filepath, "w") as f:
        # subprocess.run(["pip", "freeze"], stdout=f, text=True)
        subprocess.run(["pip-chill"], stdout=f, text=True)
    
    print(f"Pip dependents saved to {filepath}")

    return filepath

def save_list(filename, items):
    """Save a list of strings to file."""
    with open(filename, "w") as f:
        for item in sorted(items):
            f.write(item + "\n")
    print(f"Saved {filename} ({len(items)} entries)")

def get_installed_packages(timestamp, output_dir=f"{OUTPUT_DIR}/apt_mark_logs"):
    """Get dpkg package list from a Docker image."""
    cmd = f"apt-mark showmanual"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"aptmark_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    try:
        dpkg_list =  set(run_cmd(cmd).splitlines())
    except subprocess.CalledProcessError:
        dpkg_list = set()

    installed_pkgs = {line.split()[0] for line in dpkg_list if line.endswith("install")}

    save_list(filepath, installed_pkgs)

    return filepath


def save_best_model_path(best_model_path, timestamp, output_dir=f"{OUTPUT_DIR}/best_model_logs"):
    """Save the best model path into a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"best_model_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(str(best_model_path) + "\n")
    
    print(f"Best model path saved to {filepath}")
    return filepath



def arcname_from_abs(abs_path: str) -> str:
    """
    Produce an archive name that mirrors the absolute path but without the leading '/'.
    Example: '/src/a/b.txt' -> 'src/a/b.txt'
    """
    # Relativize against POSIX root so we keep the whole tree structure after '/'
    # On POSIX: relpath('/src/a', '/') -> 'src/a'
    # On Windows paths this would differ, but you’re using POSIX-like paths.
    rel = os.path.relpath(abs_path, start=os.path.sep)
    # Normalize backslashes just in case and strip leading separators
    return rel.lstrip("/\\")

def main():
    parser = argparse.ArgumentParser(description='Save for submission')

    parser.add_argument('-c', "--config", type=str, required=True, help='The config file for submission with contents to be saved in zip.')
    args = parser.parse_args()

    yaml_path = args.config
    folder, _file = os.path.split(yaml_path)
    output_zip = os.path.join(folder, "model_submission.zip")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Extract paths
    
    best_model_path = data.get("best_model_path")
    config_path = data.get("config")
    source_folder_path = "/src/models/"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pip_requirements = save_pip_packages(timestamp)
    dpkg_requirements = get_installed_packages(timestamp)
    

    
    files_to_zip = [best_model_path, config_path, source_folder_path, pip_requirements, dpkg_requirements]
    files_to_zip_with_different_dir = [{"output_path": "/src/best_model.yaml", "item": yaml_path}]

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in files_to_zip:
            if not os.path.exists(item):
                print(f"Warning: {item} not found, skipping.")
                continue

            if os.path.isdir(item):
                # Add folder recursively, preserving '/...' structure (minus the leading '/')
                for root, _, files in os.walk(item):
                    for fn in files:
                        abs_path = os.path.join(root, fn)
                        zf.write(abs_path, arcname_from_abs(abs_path))
            else:
                # Single file
                zf.write(item, arcname_from_abs(os.path.abspath(item)))

        for item_dict in files_to_zip_with_different_dir:
            item = item_dict["item"]
            output_path = item_dict["output_path"]
            if not os.path.exists(item):
                print(f"Warning: {item} not found, skipping.")
                continue
            else:
                zf.write(item, arcname_from_abs(output_path)) 

    print(f"Created {output_zip}")
  


if __name__ == "__main__":

    main()
# print(config)






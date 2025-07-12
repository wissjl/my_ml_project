import os
import subprocess

# Get the directory where run_all_jobs.py is located
# This ensures current_script_dir is always the 'cloudhoussem' directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))

models = ["lstm", "cnn_lstm", "garch_lstm"]
input_files = [f"df{i}.csv" for i in range(1, 13)] + \
              [f"eth_df{i}.csv" for i in range(1, 13)]

# Construct the full path for the output directory
output_dir = os.path.join(current_script_dir, "output_dir")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Starting model execution for {len(models)} models and {len(input_files)} input files.")
print(f"Results will be saved in: {os.path.abspath(output_dir)}\n")

# --- IMPORTANT CHANGE HERE ---
# Construct the absolute path to run_model.py
run_model_script_path = os.path.join(current_script_dir, "scripts", "run_model.py")
# -----------------------------

for model in models:
    for input_file in input_files:
        cmd = [
            "python", run_model_script_path, # <--- Use the absolute path here
            "--model", model,
            "--input_file", input_file,
            "--output_dir", output_dir
        ]
        job_name = f"{model}_{input_file.replace('.csv', '')}"
        print(f"🚀 Launching job: {job_name}")

        try:
            # --- IMPORTANT CHANGE HERE ---
            # Set the current working directory (cwd) for the subprocess
            # This ensures that run_model.py's internal relative paths (like "input_files")
            # are resolved correctly relative to the 'cloudhoussem' directory.
            result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                    cwd=current_script_dir)
            # -----------------------------
            print(f"✅ Job '{job_name}' completed successfully.")
            if result.stdout:
                print("--- STDOUT ---")
                print(result.stdout)
            if result.stderr:
                print("--- STDERR ---")
                print(result.stderr)
            print("-" * 30)

        except subprocess.CalledProcessError as e:
            print(f"❌ Job '{job_name}' failed with error code {e.returncode}.")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            print("-" * 30)
        except FileNotFoundError:
            print(f"❌ Error: 'python' command or '{run_model_script_path}' not found for job '{job_name}'.")
            print("Please ensure Python is in your PATH and scripts/run_model.py exists at the correct path.")
            print("-" * 30)
        except Exception as e:
            print(f"❌ An unexpected error occurred for job '{job_name}': {e}")
            print("-" * 30)

print("\nAll jobs launched. Check the output_dir for results and the console for any errors.")
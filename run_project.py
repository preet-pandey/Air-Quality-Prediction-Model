import subprocess
import time
import sys
import os

def run_command(command, cwd=None, name="Process", log_name=None):
    print(f"Starting {name}...")
    # On Windows with shell=True, a string is more robust
    if isinstance(command, list):
        command = " ".join(f'"{c}"' if " " in c else c for c in command)
    
    # Use a separate log file for each process to debug crashes
    if not log_name:
        log_name = f"{name.replace(' ', '_').lower()}_{int(time.time())}.log"
    
    log_file = open(log_name, "w")
    print(f"Logs for {name} will be in {log_name}")
    
    return subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )

def kill_process_on_port(port):
    print(f"Checking for existing processes on port {port}...")
    try:
        if os.name == 'nt':
            output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
            for line in output.splitlines():
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    print(f"Killing process {pid} on port {port}...")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True)
    except:
        pass

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    frontend_dir = os.path.join(base_dir, "frontend-final")

    # Clear previous logs and kill old processes
    kill_process_on_port(3000)
    kill_process_on_port(5000)
    for f in os.listdir(base_dir):
        if f.endswith(".log") and (f.startswith("flask_backend") or f.startswith("vite_frontend")):
            try:
                os.remove(os.path.join(base_dir, f))
            except:
                pass

    print("\n" + "="*50)
    print("AIR QUALITY PREDICTION PROJECT - UNIFIED RUNNER")
    print("="*50 + "\n")

    # 1. Train model
    print("Checking ML Model...")
    try:
        subprocess.run([sys.executable, "train_model.py"], cwd=backend_dir, check=True)
        print("Model is ready.\n")
    except Exception as e:
        print(f"Error training model: {e}")

    # 2. Start Backend
    backend_cmd = [sys.executable, "app.py"]
    backend_log = f"flask_backend_{int(time.time())}.log"
    backend_proc = run_command(backend_cmd, cwd=backend_dir, name="Flask Backend", log_name=backend_log)
    time.sleep(5) 

    # 3. Start Frontend
    frontend_cmd = "npm run dev -- --host 0.0.0.0 --port 3000"
    frontend_log = f"vite_frontend_{int(time.time())}.log"
    frontend_proc = run_command(frontend_cmd, cwd=frontend_dir, name="Vite Frontend", log_name=frontend_log)
    time.sleep(5)

    print("\n" + "-"*50)
    print("PROJECT IS RUNNING!")
    print(f"Dashboard: http://localhost:3000")
    print(f"Backend API: http://127.0.0.1:5000")
    print("Logs are being written to flask_backend.log and vite_frontend.log")
    print("Press Ctrl+C to stop both processes.")
    print("-"*50 + "\n")

    try:
        while True:
            if backend_proc.poll() is not None:
                print(f"ERROR: Backend process exited with code {backend_proc.returncode}.")
                # Print last few lines of log
                with open(backend_log, "r") as f:
                    print("Backend Log Tail:\n", f.readlines()[-5:])
                break
            if frontend_proc.poll() is not None:
                print(f"ERROR: Frontend process exited with code {frontend_proc.returncode}.")
                with open(frontend_log, "r") as f:
                    print("Frontend Log Tail:\n", f.readlines()[-5:])
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        if 'backend_proc' in locals(): backend_proc.terminate()
        if 'frontend_proc' in locals(): frontend_proc.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()

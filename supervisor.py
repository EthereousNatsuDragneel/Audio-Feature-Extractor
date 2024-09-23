import subprocess
import time
import os
import signal

def run_a_py():
	return subprocess.Popen(['python3', 'features_extractor.py'])

def monitor_a_py(proc):
	try:
		while True:
			# Check if the process is still running
			retcode = proc.poll()
			if retcode is not None:
				# Process has terminated
				print(f'features_extractor.py terminated with return code {retcode}. Restarting...')
				return False
			time.sleep(1)
	except KeyboardInterrupt:
		print('supervisor.py received a termination signal. Exiting...')
		proc.terminate()
		return True

def main():
	while True:
		proc = run_a_py()
		should_exit = monitor_a_py(proc)
		if should_exit:
			break
		# Ensure the process has truly terminated before restarting
		proc.wait()
		print("a was killed")

if __name__ == "__main__":
	main()

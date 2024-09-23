import sqlite3
import os
import fnmatch
from segmenter import Segmenter
import sys
from pydub import AudioSegment
import shutil

def delete_table(database_path, table_name):
	"""
	Deletes a table from the SQLite database.
	
	Parameters:
	database_path (str): Path to the SQLite database file.
	table_name (str): Name of the table to be deleted.
	"""
	try:
		# Connect to the SQLite database
		conn = sqlite3.connect(database_path)
		cursor = conn.cursor()
		
		# Create the SQL command to drop the table
		sql_command = f"DROP TABLE IF EXISTS {table_name};"
		
		# Execute the SQL command
		cursor.execute(sql_command)
		
		# Commit the changes
		conn.commit()
		
		print(f"Table '{table_name}' has been deleted successfully.")
	except sqlite3.Error as e:
		print(f"An error occurred: {e}")
	finally:
		# Close the connection
		if conn:
			conn.close()

# Example usage:
# delete_table("your_database.db", "your_table_name")

def delete_until_target_file(target_file, root_dir="/mnt/d/ubc/miles/r/"):
	# Walk the directory tree
	for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
		# If the target file is found, stop deletion
		if target_file in filenames:
			print(f"Target file '{target_file}' found. Stopping deletion.")
			file_path = os.path.join(dirpath, target_file)
			os.remove(file_path)
			return
		
		# Delete files in the current directory
		for filename in filenames:
			file_path = os.path.join(dirpath, filename)
			print(f"Deleting file: {file_path}")
			os.remove(file_path)
		
		# Delete the current directory if it is empty
		if not os.listdir(dirpath):
			print(f"Deleting directory: {dirpath}")
			os.rmdir(dirpath)
	print(f"Target file '{target_file}' not found in the directory tree.")

def db_corpus_sync():
	x = print_last_row_from_sample_table("test.db")[0]

	# Construct the filename
	filename = f"/mnt/d/ubc/miles/corpus/{x}.wav"
	
	# Check if the file exists
	if os.path.isfile(filename):
		return x+1
	else:
		delete_row_by_id("test.db", x)
		return x

def read_file_until_wav(file_path="last.txt"):
	with open(file_path, 'r') as file:
		line = file.readline().strip()
		
		# Find the index of ".wav" in the string
		wav_index = line.find('.wav')
		
		# Check if ".wav" is found in the string
		if wav_index == -1:
			raise ValueError('The file does not contain ".wav".')
		
		# Return the substring up to and including ".wav"
		result = line[:wav_index + 4]
		result = result.rsplit('/', 1)[-1]
		return result

def delete_row_by_id(db_file, id_to_delete):
	# Connect to the SQLite database
	conn = sqlite3.connect(db_file)
	cursor = conn.cursor()

	try:
		# Delete the row with the provided ID from the "sample" table
		cursor.execute("DELETE FROM sample WHERE id=?", (id_to_delete,))
		conn.commit()
		print(f"Row with ID {id_to_delete} deleted successfully.")
	except sqlite3.Error as e:
		print("Error deleting row:", e)
	finally:
		# Close the cursor and connection
		cursor.close()
		conn.close()

def print_last_row_from_sample_table(db_file):
	# Connect to the SQLite database
	conn = sqlite3.connect(db_file)
	cursor = conn.cursor()

	try:
		# Fetch the last row from the "sample" table
		cursor.execute("SELECT * FROM sample ORDER BY ROWID DESC LIMIT 1")
		row = cursor.fetchone()

		if row:
			return(row)
		else:
			print("Table 'sample' is empty.")
	except sqlite3.Error as e:
		print("Error reading data:", e)
	finally:
		# Close the cursor and connection
		cursor.close()
		conn.close()

def print_ids_from_db(db_file):
	# Connect to the SQLite database
	conn = sqlite3.connect(db_file)
	cursor = conn.cursor()

	try:
		# Fetch all values from the "id" column in the "sample" table
		cursor.execute("SELECT id FROM sample")
		rows = cursor.fetchall()

		# Print all values in the "id" column
		for row in rows:
			print(row[0])  # Assuming "id" is the first (0-indexed) column
	except sqlite3.Error as e:
		print("Error reading data:", e)
	finally:
		# Close the cursor and connection
		cursor.close()
		conn.close()

def record_progress(line_of_text):
	"""
	This is only a helper function to record the last file that was processed by the corpus maker, so that if the program has to be killed we can resume from where we left off
	It will write the last file the program worked on, as well as how many files have been worked on (for id purposes)
	 Example usage
	record_progress("/mnt/d/ubc/miles/audio/93.wav,93")
	"""
	file_path = "last.txt"	
	with open(file_path, 'w') as file:
		file.write(line_of_text)

def extract_segment(audiofile_location, start, end, output_location):
	"""
	This function takes params as the input audio file (wav), start (time in ms), end (ms), output file path and name (wav)
	It will take the input audio and export the specified audio segment as a wav file
	 Example usage:
	extract_segment("/mnt/d/ubc/miles/x.wav", 5000, 11000, "/mnt/d/ubc/miles/output.wav")
	"""
	audio = AudioSegment.from_wav(audiofile_location)
	segment = audio[start:end]
	segment.export(output_location, format="wav")

def process_files(directory, start_file=None, i=1, limit=10):
	l=i
	s = Segmenter()
	start_processing = not bool(start_file)  # Start processing if start_file is None
	for root, dirs, files in os.walk(directory):
		for name in files:
			file_path = os.path.join(root, name)
			if fnmatch.fnmatch(name, '*.wav'):
				if not start_processing:
					if file_path == start_file:
						start_processing = True
					continue

				#print(f"Found wav file: {file_path}")
				try:
					val = s.extract_stable_segment(file_path)
					if val is not None:
						insert_values("test.db", "sample", [(i, file_path, val['start'], val['end'], val['type'])])
						extract_segment(file_path, val['start']*1000, val['end']*1000, "/mnt/d/ubc/miles/corpus/"+str(i)+".wav")
						record_progress(file_path+","+str(i))
						i = i+1
						counter = counter+1
						os.system('clear')
						if limit>i-l:
							return
				except Exception as e:
					print("an exception occured for an audio file: ",e)
			else:
				# Ignore other file types
				pass
	print(i)

"""
# Example usage:
directory_path = '/path/to/your/directory'
process_files(directory_path)
"""

def create_table(db_name, table_name, schema):
	"""
	Create a table with the given schema.

	:param db_name: Name of the database file.
	:param table_name: Name of the table to create.
	:param schema: Dictionary of column names and types.
	"""
	conn = sqlite3.connect(db_name)
	cursor = conn.cursor()

	# Create the table
	columns = ', '.join([f"{col} {col_type}" for col, col_type in schema.items()])
	cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
	
	conn.commit()
	conn.close()

def insert_values(db_name, table_name, values):
	"""
	Insert values into the specified table.

	:param db_name: Name of the database file.
	:param table_name: Name of the table to insert into.
	:param values: List of tuples, where each tuple contains the values for one row.
	"""
	conn = sqlite3.connect(db_name)
	cursor = conn.cursor()

	# Get column count
	cursor.execute(f"PRAGMA table_info({table_name})")
	columns = cursor.fetchall()
	column_count = len(columns)

	# Construct the placeholders
	placeholders = ', '.join(['?' for _ in range(column_count)])

	# Insert values
	cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", values)
	
	conn.commit()
	conn.close()

"""
# Example usage
if __name__ == "__main__":
	db_name = 'example.db'
	table_name = 'example_table'
	schema = {
		'id': 'INTEGER PRIMARY KEY',
		'name': 'TEXT',
		'age': 'INTEGER'
	}

	# Create the table
	create_table(db_name, table_name, schema)

	# Insert values
	values_to_insert = [
		(1, 'Alice', 30),
		(2, 'Bob', 25),
		(3, 'Charlie', 35)
	]
insert_values(db_name, table_name, values_to_insert)
"""

#process_files("D:/UBC/miles/audio")

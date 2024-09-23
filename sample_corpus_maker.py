from corpus_maker import *
import signal
import sys

def signal_handler(sig, frame):
	print('kill order received')
	# Perform any cleanup or final actions here
	sys.exit(0)

# Register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

schema = {
	'id': 'INTEGER PRIMARY KEY',
	'path': 'TEXT',
	'start': 'INTEGER',
	'end': 'INTEGER',
	'class': 'TEXT'}

#create_table("test.db", "sample", schema)

index = db_corpus_sync()
delete_until_target_file(read_file_until_wav())
process_files("/mnt/d/ubc/miles/r/", i=index)


#for i in range(50):
#	delete_row_by_id("test.db", 2200-i)
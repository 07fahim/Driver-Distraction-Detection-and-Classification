# gunicorn.conf.py
import multiprocessing

workers = 2
worker_class = 'sync'
timeout = 300  # 5 minutes for video processing
keepalive = 5
max_requests = 100
max_requests_jitter = 10

# Use RAM for temp files
worker_tmp_dir = '/dev/shm'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
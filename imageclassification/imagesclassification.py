import  os, sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
from keras.models import load_model
import numpy as np
import cv2
import pyperclip
import time
import signal
from datetime import datetime
from threading import Thread, Event
sys.stderr = stderr
from tqdm import tqdm

#These are the only things that should need to change
res1 = 80		#The resolution that we want the images to be
res2 = 60
freq=25			#The number of frames you want to try to process per second
workers=4		#The number of threads that will be active at one time
runtime=3600	#The number of seconds the program will run
#Runtime is only here for safety because force closing a multithreaded process is never a good idea

#Basic nupimber crunching, initialization, and directory walking to get everything prepared to launch
default_dir = '/home/yasser/Desktop/imageclasses'
for (_,_,files) in os.walk(default_dir):
	break
for file in files:
	if file.endswith('.h5'):
		global model
		model=load_model(default_dir+'/'+file)
		i=np.zeros([1, res1, res2, 1])*255
		model.predict_classes(i)
		print('\n The model was actually loaded \n')
		break
cap=cv2.VideoCapture(0)
cap.set(3,res1)
cap.set(4,res2)
try:
	ready,img=cap.read()
except:
	pass
translator=np.array(['L','N','R','U'])
pbar=tqdm(total=0, bar_format='  Time: {elapsed}  Rate: {rate_fmt}  Total: {n}  Output: {desc}', unit=" frame",mininterval=0.001,maxinterval=1.)
w=1/freq/workers
period=1/freq
threads = []
if not cap.isOpened():
	print('\n The VideoCapture object is closed \n')
finish_time=time.time()+runtime

'''
#ReusableThread makes threads reusable, saving the overhead of allocating space for new threads 20 times per second
class ReusableThread(Thread):
	def __init__(self, target):
		self._startSignal = Event()
		self._oneRunFinished = Event()
		self._finishIndicator = False
		self._callable = target
		Thread.__init__(self)
	def restart(self):
		self._startSignal.set()
	def run(self):
		self.restart()
		while(True):	
			self._startSignal.wait()
			self._startSignal.clear()
			if(self._finishIndicator):
				self._oneRunFinished.set()
				return
			self._callable()
			self._oneRunFinished.set()
	def join(self):
		self._oneRunFinished.wait()
		self._oneRunFinished.clear()
	def bounce(self):
		self._oneRunFinished.wait()
		self._oneRunFinished.clear()
		self._startSignal.set()
	def finish(self):
		self._finishIndicator = True
		self.restart()
		self.join()
'''
#The function everything is built around
def img_process():
	ready,img=cap.read()
	global model
	img = cv2.resize(img,(res2,res1),interpolation = cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.expand_dims(img,axis=0)
	img = np.expand_dims(img,axis=-1)
	#cv2.imshow('img', img)
	p = translator[model.predict_classes(img/255)[0]]
	pyperclip.copy(p)
	#print(p)
	pbar.update()
	pbar.set_description_str("%s" % p)
	time.sleep(0.05)

#The legendary handler that allows all threads to shutdown gracefully on a CTRL+C force close
'''def signal_handler(signal, frame):
	for i in range(workers):
		threads[i].finish()
	pbar.close()
	cap.release()
	sys.exit(0)

#Load up all of our threads into the list
for _ in range(workers):
	threads.append(ReusableThread(target = img_process))
signal.signal(signal.SIGINT, signal_handler)

#Startup the threads in even intervals 
for i in range(workers):
	threads[i].daemon=True
	threads[i].start()
	time.sleep(w)
	
#Join and restart the threads at the desired frequency 
while time.time()<finish_time:
	for i in range(workers):
		threads[i].bounce()
		time.sleep(period)

#Cleanup the threads, video capture object, and progress bar formatter
for i in range(workers):
	threads[i].finish()
'''
while (True):
	img_process()
	time.sleep(0.1)

pbar.close()
cap.release()



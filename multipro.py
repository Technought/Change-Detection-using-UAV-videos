import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.messagebox as tkMessageBox
from multiprocessing import Process, Queue, Manager
from queue import Empty

DELAY1 = 80
DELAY2 = 20

class 
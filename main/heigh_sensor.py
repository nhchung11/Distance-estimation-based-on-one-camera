import threading
import serial
import time

# Create a class that extends Thread
class UARTThread(threading.Thread):
    def __init__(self, port, baudrate):
        threading.Thread.__init__(self)
        self.ser = serial.Serial(port, baudrate)
        self.daemon = True

    def run(self):
        while True:
            # Read from UART device
            if self.ser.in_waiting > 0:
                data = self.ser.readline()
                print("Received data: ", data)
            time.sleep(1)  # sleep for 1 second

import threading
import smbus
import time

# Create a class that extends Thread
class I2CThread(threading.Thread):
    def __init__(self, bus_number, device_address):
        threading.Thread.__init__(self)
        self.bus = smbus.SMBus(bus_number)
        self.address = device_address
        self.daemon = True

    def run(self):
        while True:
            # Read from I2C device
            data = self.bus.read_byte(self.address)
            print("Received data: ", data)
            time.sleep(1)  # sleep for 1 second

import serial
import datetime
import csv
import math
import sys # read in line argument
import time # end streaming after the video nows
import datetime

forcemeter_port = "COM5"#"/dev/ttyS3"
baud = 2400
ser_ForceMeter = serial.Serial(forcemeter_port, baud,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
ser_arduino = serial.Serial("COM3",9600)
row_data = []
print("Connected to Force Meter:" + forcemeter_port)
print("Connected to Arduino:" + "COM3")

start = time.time()
if len(sys.argv) == 3:#scriptname + No.material (1-11) + trial (0~5)
    fileName =  "./"+str(sys.argv[1])+"/data_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".csv"
    force_level = 10
    trial = int(sys.argv[2])
    if trial == 0:
        duration = 10
    else:
        duration = 20
    print("Writing "+fileName)
else:
    fileName =  "data_"+"test_10.csv"
    duration = math.inf
    print("Writing "+fileName)

with open(fileName, "w") as target_file:
    writer = csv.writer(target_file)
    now = time.time()
    previous_data = 0.0
    ser_arduino.write(b'D')
    while ((now-start)<duration):
        data = ser_ForceMeter.read(6)
        data = float(data.decode('UTF-8','ignore'))
        current_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')[:-3]
        # print(current_time+"  "+str(data)) # display the data to the terminal
        now = time.time()
        if (previous_data >= -force_level and data < -force_level):
            ser_arduino.write(b'U')
        row_data = [current_time, data]
        print(row_data)
        writer.writerow(row_data) # write data with a newline
        previous_data = data
 
ser_ForceMeter.close()
ser_arduino.write(b'U')
ser_arduino.close()
print(str(duration)+" seconds have passed. Streaming stopped.")
print("Data Written in "+ fileName)
    

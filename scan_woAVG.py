import pywifi
import time
import csv
from datetime import datetime
import os


def scan_aps(location_name= "name", run_number= 1):
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0] #first wifi adapter
    print(f"Interface: {iface.name()}")

    all_scans = [] #store all readings across scans

    #scan for nearby access points three times
    for i in range(20):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"scan {i+1} for {location_name}")
        iface.scan()
        time.sleep(5)

        results = iface.scan_results() #lists the detected networks
        print(f"Found {len(results)} networks in this scan")

        for network in results:
            ssid = network.ssid if network.ssid else "Hidden_Network" #no name
            signal = network.signal

            all_scans.append([ssid, signal, timestamp, location_name, i, run_number])

        
    #save in a csv
    try:
        file_exists = os.path.exists('Library_woAVG.csv')
        mode = 'a' if file_exists else 'w'

        with open('Library_woAVG.csv', mode, newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(['SSID', 'Signal(dBm)', 'Timestamp', 'Location', 'Scan_ID', 'Run']) #header row
            
            writer.writerows(all_scans)

        print("✅Stored data successfully")
    except Exception as e:
        print(f"an error occurred while writing the csv: {e}")

if __name__ == "__main__":
    scan_aps("Stairs/RightEnd", 5)
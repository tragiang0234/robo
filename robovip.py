import cv2
import numpy as np
import serial
import time

# Initialize serial communication with Arduino
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)  # Adjust 'COM6' to your port
time.sleep(2)  # Wait for the connection to establish

# Initialize webcam
vid = cv2.VideoCapture(0)  # Adjust the camera index if necessary

# Verify if the camera opened successfully
if not vid.isOpened():
    print("Error: Could not access the camera.")
    exit()

# PID controller variables
kp = 0.5  # Proportional gain
ki = 0.01  # Integral gain
kd = 0.1  # Derivative gain
previous_error = 0
integral = 0

# Distance thresholds
GRIP_DISTANCE = 7.0  # Distance in cm to trigger gripping
SLOW_DISTANCE = 10.0  # Distance in cm to reduce speed

def send_command(command):
    """Send a command to the Arduino."""
    arduino.write(command.encode())
    print(f"Command Sent: {command}")

def slow_rotation_and_detection():
    """Slowly rotate the robot while allowing the camera to detect green objects."""
    print("Starting slow rotation for detection...")
    send_command("L")  # Rotate left slowly (can be adjusted to "R" for right)
    
    # Give some time for the robot to rotate while detecting green color
    for _ in range(10):  # Rotate for 10 iterations (can adjust the number of steps)
        ret, frame = vid.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Get the frame dimensions
        height, width, _ = frame.shape

        # Convert the frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for green color in HSV
        lower_green = np.array([35, 100, 100])  # Adjust as needed
        upper_green = np.array([85, 255, 255])  # Adjust as needed

        # Create a mask to extract green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_detection = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours of the green object
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (assumes it's the green object)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the green object
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw a rectangle and circle at the center of the green object
            cv2.rectangle(green_detection, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(green_detection, (center_x, center_y), 5, (0, 0, 255), -1)

            # Calculate the error between the object center and the frame center
            error = center_x - width // 2

            # PID calculation
            integral += error
            derivative = error - previous_error
            correction = kp * error + ki * integral + kd * derivative
            previous_error = error

            # Simulate distance check using ultrasonic sensor
            send_command("D")  # Ask Arduino for distance
            distance = arduino.readline().decode().strip()  # Read distance from Arduino

            if distance:
                try:
                    distance = float(distance)
                    print(f"Distance: {distance} cm")

                    # Adjust robot behavior based on distance
                    if distance <= GRIP_DISTANCE:
                        send_command("S")  # Stop the robot
                        send_command("G")  # Trigger gripping mechanism
                        return  # Stop after gripping
                    elif distance <= SLOW_DISTANCE:
                        send_command("SF")  # Slow forward
                    else:
                        if abs(error) < 10:  # If the object is near the center
                            send_command("F")  # Move forward
                        elif correction > 0:
                            send_command("R")  # Turn right
                        else:
                            send_command("L")  # Turn left

                except ValueError:
                    print("Error: Invalid distance value received.")

        # Wait a bit before next rotation step (to control the rotation speed)
        time.sleep(0.1)  # Adjust as necessary

    send_command("S")  # Stop after rotating

# Initial slow rotation for camera detection
slow_rotation_and_detection()

while True:
    # Capture frame from the webcam
    ret, frame = vid.read()
    if not ret:
        print("Error: Failed to capture frame.")
        continue  # Skip this iteration if frame capture failed

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([35, 100, 100])  # Adjust as needed
    upper_green = np.array([85, 255, 255])  # Adjust as needed

    # Create a mask to extract green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_detection = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours of the green object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (assumes it's the green object)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center of the green object
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw a rectangle and circle at the center of the green object
        cv2.rectangle(green_detection, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(green_detection, (center_x, center_y), 5, (0, 0, 255), -1)

        # Calculate the error between the object center and the frame center
        error = center_x - width // 2

        # PID calculation
        integral += error
        derivative = error - previous_error
        correction = kp * error + ki * integral + kd * derivative
        previous_error = error

        # Simulate distance check using ultrasonic sensor
        send_command("D")  # Ask Arduino for distance
        distance = arduino.readline().decode().strip()  # Read distance from Arduino

        if distance:
            try:
                distance = float(distance)
                print(f"Distance: {distance} cm")

                # Adjust robot behavior based on distance
                if distance <= GRIP_DISTANCE:
                    send_command("S")  # Stop the robot
                    send_command("G")  # Trigger gripping mechanism
                    break  # Stop loop after gripping
                elif distance <= SLOW_DISTANCE:
                    send_command("SF")  # Slow forward
                else:
                    if abs(error) < 10:  # If the object is near the center
                        send_command("F")  # Move forward
                    elif correction > 0:
                        send_command("R")  # Turn right
                    else:
                        send_command("L")  # Turn left

            except ValueError:
                print("Error: Invalid distance value received.")

    # Display the green detection frame (Optional, uncomment to visualize)
    # cv2.imshow("Green Detection", green_detection)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        send_command("S")  # Stop the robot
        break

# Release resources
vid.release()
cv2.destroyAllWindows()
arduino.close()
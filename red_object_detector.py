import cv2
import numpy as np

def detect_red_objects():
    # Initialize video capture from default camera (0)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define red color range in HSV
        # Red wraps around in HSV, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.add(mask1, mask2)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and count objects
        red_objects_count = 0
        min_area = 500  # Minimum area to consider as an object
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                red_objects_count += 1
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Display count on frame
        cv2.putText(frame, f'Red Objects: {red_objects_count}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Red Object Counter', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_red_objects() 
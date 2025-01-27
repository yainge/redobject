import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class RedObjectDetector:
    def __init__(self):
        try:
            self.model = load_model('red_object_model.h5')
            self.class_indices = np.load('class_indices.npy', allow_pickle=True).item()
            self.categories = {v: k for k, v in self.class_indices.items()}
        except:
            print("Warning: Model files not found. Running in detection-only mode.")
            self.model = None
            self.categories = None

    def preprocess_roi(self, roi):
        # Preprocess region of interest for the model
        roi_resized = cv2.resize(roi, (64, 64))
        roi_normalized = roi_resized / 255.0
        return np.expand_dims(roi_normalized, axis=0)

    def classify_object(self, roi):
        if self.model is None:
            return "Unknown"
        
        preprocessed_roi = self.preprocess_roi(roi)
        prediction = self.model.predict(preprocessed_roi, verbose=0)
        predicted_class = self.categories[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return f"{predicted_class} ({confidence:.1f}%)"

    def detect_and_classify_red_objects(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define red color range in HSV
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.add(mask1, mask2)
            
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            red_objects_count = 0
            min_area = 500
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    red_objects_count += 1
                    
                    # Get bounding box for the object
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = frame[y:y+h, x:x+w]
                    
                    # Classify the object
                    classification = self.classify_object(roi)
                    
                    # Draw contour and classification
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    cv2.putText(frame, classification, 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (0, 255, 0), 2)
            
            cv2.putText(frame, f'Red Objects: {red_objects_count}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0), 2)
            
            cv2.imshow('Red Object Counter & Classifier', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = RedObjectDetector()
    detector.detect_and_classify_red_objects() 
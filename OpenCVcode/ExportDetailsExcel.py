import pandas as pd

# Extracting the details of the detected objects
boxes = results[0].boxes  # Bounding box coordinates
confidences = boxes.conf  # Confidence scores
class_ids = boxes.cls  # Class IDs
class_names = results[0].names  # Class names

# Prepare data for the DataFrame
data = []

# Loop through the boxes and extract the details
for box, confidence, class_id in zip(boxes.xyxy, confidences, class_ids):
    # Get the class name based on the class ID
    class_name = class_names[int(class_id)]
    
    # Create a dictionary for each detection
    detection = {
        'Class Name': class_name,
        'Confidence': confidence.item(),
        'x1': box[0].item(),
        'y1': box[1].item(),
        'x2': box[2].item(),
        'y2': box[3].item()
    }
    
    # Add to the data list
    data.append(detection)

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Define the output Excel file path
output_excel_path = "C:/Users/dolap/Downloads/detected_objects.xlsx"

# Export the DataFrame to an Excel file
df.to_excel(output_excel_path, index=False)

print(f"Excel file saved to {output_excel_path}")

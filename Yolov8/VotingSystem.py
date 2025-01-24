import pandas as pd
from collections import defaultdict

# Extracting the details of the detected objects
boxes = results[0].boxes  # Bounding box coordinates
confidences = boxes.conf  # Confidence scores
class_ids = boxes.cls  # Class IDs
class_names = results[0].names  # Class names

# Create a dictionary to track the total confidence for each class
confidence_sum = defaultdict(float)

# Loop through the boxes and accumulate the confidence score for each class
for box, confidence, class_id in zip(boxes.xyxy, confidences, class_ids):
    # Get the class name based on the class ID
    class_name = class_names[int(class_id)]
    
    # Add the confidence score to the total confidence for that class
    confidence_sum[class_name] += confidence.item()

# Calculate the total confidence across all classes
total_confidence = sum(confidence_sum.values())

# Create a list of classes and their percentage of total confidence
vote_data = []
for class_name, total_conf in confidence_sum.items():
    # Calculate the percentage of total confidence for each class
    percentage = (total_conf / total_confidence) * 100
    vote_data.append({'Class Name': class_name, 'Confidence (%)': percentage})

# Create a DataFrame from the voting data
df_votes = pd.DataFrame(vote_data)

# Sort the DataFrame by 'Confidence (%)' in descending order
df_votes = df_votes.sort_values(by='Confidence (%)', ascending=False)

# Define the output Excel file path for the voting results
output_vote_excel_path = "C:/Users/dolap/Downloads/voting_results_with_percentage.xlsx"

# Export the DataFrame to an Excel file
df_votes.to_excel(output_vote_excel_path, index=False)

print(f"Voting results with confidence percentages saved to {output_vote_excel_path}")


#To visualize the voting results
import matplotlib.pyplot as plt

# Create a bar plot of the voting results
plt.figure(figsize=(10, 6))
plt.bar(df_votes['Class Name'], df_votes['Confidence (%)'], color='skyblue')
plt.xlabel('Class Name')
plt.ylabel('Confidence (%)')
plt.title('Voting System: Detected Classes with Confidence')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# To handle multiple frams in a video
# Initialize a global confidence dictionary for all frames
confidence_sum = defaultdict(float)

# Loop through video frames
for frame in frames:
    results = model(frame)  # Run inference on the frame
    # Update the global confidence dictionary for this frame
    for box, confidence, class_id in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        class_name = results[0].names[int(class_id)]
        confidence_sum[class_name] += confidence.item()

# After processing all frames, calculate percentages and export to Excel as shown before
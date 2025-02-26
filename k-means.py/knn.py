def knn(feature, single_test_input, k):
    X_train['distance'] = abs(X_train[feature] - single_test_input[feature])
    prediction = y_train[X_train["distance"].nsmallest(n=k).index].mode()[0]
    return prediction

model_prediction = knn("age", X_test.iloc[108], 3)
print(f"Predicted label: {model_prediction}")
print(f"Actual label: {y_test.iloc[108]}")

#Create a function knn() that has the following parameters:
# feature: the feature we want to use to calculate the distance.
# single_test_input: a data point from the test set.
# k: the number of neighbors.
# Inside the function:
# Calculate the Euclidean distance between the single_test_input and every observation in X_train for the given feature. Save the distances in a new column, distance, in X_train.
# For the k rows in distance with the smallest distance values, identify the most common label for the same rows in y_train. Save the label to the variable prediction.
# Return prediction.
# Call the function, knn() with the following arguments:
# feature = "age".
# For single_test_input, select the row at index 108 from X_test.
# k = 3.
# Print the output of the above step.
# Print the true label (y_test) corresponding to the single_test_input used above.
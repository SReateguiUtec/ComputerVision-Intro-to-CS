# Computer Vision Digit Recognition Project

This project implements a digit recognition system that identifies handwritten digits from images by calculating Euclidean distances. The system compares input images against the sklearn digits dataset to find the most similar digits.

## Features

- Interactive menu to display average images of digits 0-9
- Custom digit recognition from user-provided images
- Comparison of user images against sklearn digits dataset
- Three closest digit matches with distance measurements
- Classification based on nearest neighbors

## Requirements

- Python 3.x
- NumPy
- Scikit-learn
- Matplotlib
- OpenCV (cv2)
- Pillow (PIL)

Install the required packages using pip:

```bash
pip install numpy scikit-learn matplotlib opencv-python pillow
```

## Usage

1. Place your digit image file (e.g., `5.png`) in the project directory
2. Run the script: `Python3 ProjectSZN.py`
3. Use the interactive menu to:
   - Display all average digit images (option #1)
   - Display a specific digit (option #2)
   - Exit the menu (option #3)

After exiting the menu, the system will:
1. Load and process your custom digit image
2. Resize it to 8x8 pixels
3. Compare it against the sklearn digits dataset
4. Display the 3 closest matches with distances
5. Classify the digit based on the nearest neighbors
6. Show the closest average digit image match

## How It Works

1. **Dataset Processing**: The system loads the sklearn digits dataset and calculates average images for each digit (0-9)
2. **Image Preprocessing**: Custom images are resized to 8x8 pixels and pixel values are adjusted for better recognition
3. **Distance Calculation**: Euclidean distance is used to compare the input image with all digits in the dataset
4. **Classification**: The system identifies the three closest matches and determines the most likely digit based on nearest neighbors

## Project Structure

- `ProjectSZN.py`: Main script containing all functionality
- Input image (e.g., `5.png`): Custom digit image for recognition

## Output

The system provides:
- Visual display of average digit images
- ASCII art representation of the input image
- Numerical array of the processed image
- Three closest digit matches with distances
- Final classification result
- Closest average digit match

## Note

This project was created as part of a Introduction to Computer Science course in my freshman year.
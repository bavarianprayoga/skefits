# Skefits
*Skefits* is a fashion recommendation system that uses CNN models and K-Nearest Neighbor algorithm to give outfit recommendations based on provided images using camera or image files. This is one of my first projects.

I used `MobileNetV3` as i initially planned to deploy it on mobile. But due to my limitation on Flutter. Deploying it on web using Streamlit will do for now. `ResNet` will also be an option. 

## Dataset Used
This model uses [Param Aggarwal's fashion dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) with roughly 44k images, 15GB in size including metadata and individual JSON. It is quite old for fast-growing fashion scene. But the amount of content contained in this dataset is rich. Especially the JSON, and that's what i'm looking for.

## Limitations and Challenges
One of the things that i haven't fix is the webcam freezing when deploying. It still can take an image, but it only shows the first frame it took and froze.

The main challenges when building this project was getting the Flutter app (Android in this case) to connect with the model and give expected output. As i have little to no knowledge about android development. So, until i got more familiar with it, i will leave it as it is.

## Future Implementations
In the future, i planned to implement more complex algorithm to not just looking for similarities. Using Vision Transformers could be one. But i'll need bigger dataset and more computational power.

## Project Content
When training, i've decided to extract images features from the dataset so we don't need to do it everytime we deploy.
```
fashion_model.keras - The main model file
image_features.npy - contains pre-computed features for similarity search
neighbor_index.joblib - the search index for finding similar items
label_encoders.pkl - for decoding predictions into categories
valid_indices.npy - maps search results back to original indices
metadata.csv - The dataset metadata, used for getting images properties
```

## Installation and Usage
I used Windows 11 and Python 3.11.9 when building this project. I don't know how would it run on other machines. To run, simply install requirements.txt provided in the project file or build a venv. That's up to you.
```
pip install -r requirements.txt
```
Then run
```
streamlit run app.py
```
I have also provided the training source code called skefits_train.ipynb. Feel free to use it, thanks Claude.

## Screenshots


## Built With
- [Python (3.11.9)](https://www.python.org/) - Core programming language
- [OpenCV](https://opencv.org/) - Camera interface and image processing
- [TensorFlow](https://www.tensorflow.org/) - Deep learning and feature extraction
- [Streamlit](https://streamlit.io/) - Web interface and application framework
- [NumPy](https://numpy.org/) - Numerical computing and array operations
- [Pandas](https://pandas.pydata.org/) - Data manipulation and metadata handling
- [Joblib](https://joblib.readthedocs.io/en/stable/) - Model and data persistence
- [Pillow](https://pypi.org/project/pillow/) - Additional image processing support

# Conclusion
For my first project, i think i did decent. While there's a lot to fix and improve, doing this project gained plenty knowledge about CNN models, Android development, and algorithms. I hope that this project will be a foundation for my career as a programmer.

# face recognition
This face recognition implementation is capable of recognizing faces with a certain level of occlusion, this includes faces wearing masks.
You can also add new users manually by adding a photo in the images folder.


# How to run:
<pre><code>python main.py --input webcam</code></pre>

![alt text](https://github.com/juan-csv/face_recognition_occlusion/blob/master/results/result.gif)

if you don't want to run it with the webcam use

<pre><code>python main.py --input image --path_im test_image.jpeg</code></pre>

![alt text](https://github.com/juan-csv/face_recognition_occlusion/blob/master/results/test.jpg)

# Add new faces to the database (facial recognition)
You can add new users to the faces database simply by adding the person's photo in format .jpg in the **images** folder, for the registry to work correctly, only the person of interest should appear in the photo.

# References

- **Face recognition:** https://github.com/ageitgey/face_recognition



single-user face recognition  using Google Cloud Vision for face detection
and landmark feature extraction. Known-face encodings are computed from face landmarks
and attributes and stored locally. New images or webcam frames are sent to Vision, encoded,
and compared to known encodings using cosine similarity.

Requirements (install instructions below):
 - python3.8+
 - google-cloud-vision
 - opencv-python
 - pillow
 - numpy

Usage examples:
 - Build/refresh encodings from known images:
     python face_mvp.py --known-dir ./known_faces --encodings encodings.npz --credentials /path/key.json --build
 - Recognize a single image:
     python face_mvp.py --input test.jpg --encodings encodings.npz --credentials /path/key.json
 - Run webcam (press 'q' to quit, 's' to force-check current frame):
   
     python face_mvp.py --input webcam --encodings encodings.npz --credentials /path/key.json

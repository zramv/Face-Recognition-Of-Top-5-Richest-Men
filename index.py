import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(480, 480))


known_embeddings = {}
threshold = 0.5 

known_faces_dir = "known_faces"

for person_name in os.listdir(known_faces_dir):
    person_folder = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    embeddings_list = []

    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path)

            faces = app.get(img)
            if len(faces) > 0:
                emb = faces[0].normed_embedding
                embeddings_list.append(emb)
                print(f"Added embedding for {person_name} from {filename}")
            else:
                print(f"No face detected in {filename}")

    if embeddings_list:
        known_embeddings[person_name] = embeddings_list

print("Loaded embeddings for persons:", list(known_embeddings.keys()))

# ---------------------------------------------
# Open video file instead of webcam
# ---------------------------------------------

video_path = "test_video/Mark Zuckerberg Video.mp4"   # replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect all faces in the frame
    faces = app.get(frame)

    for face in faces:
        emb = face.normed_embedding

        name = ""
        max_sim = 0
        color = (0, 255, 0)
        for person_name, person_embeddings in known_embeddings.items():
            # Compute cosine similarity between detected face and all images of this person
            sims = cosine_similarity([emb], person_embeddings)
            best_sim = np.max(sims)

            if best_sim > max_sim and best_sim > threshold:
                name = person_name
                max_sim = best_sim
                color = (255,0,0)

        # Draw bounding box around the face
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"{name}", (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
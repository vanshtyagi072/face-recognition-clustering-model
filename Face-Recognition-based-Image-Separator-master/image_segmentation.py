import os
import cv2
import pickle
import numpy as np
import face_recognition

# Save encodings
def saveEncodings(encs, names, fname="encodings.pickle"):
    data = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
    
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    print("[INFO] Encodings saved to", fname)

# Read encodings
def readEncodingsPickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    encodings = [d["encoding"] for d in data]
    names = [d["name"] for d in data]
    return encodings, names

# Create encodings
def createEncodings(image):
    face_locations = face_recognition.face_locations(image)
    known_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return known_encodings, face_locations

# Compare encodings
def compareFaceEncodings(unknown_encoding, known_encodings, known_names):
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return True, known_names[best_match_index], face_distances[best_match_index]
    else:
        return False, "", face_distances[best_match_index]

# Save image to directory
def saveImageToDirectory(image, name, imageName):
    path = os.path.join("C:/Users/Admin/Desktop/dataset", name)
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, imageName), image)

# Process known people images
def processKnownPeopleImages(path="C:/Users/Admin/Desktop/people", saveLocation="./known_encodings.pickle"):
    known_encodings = []
    known_names = []
    
    if not os.path.isdir(path):
        print(f"[ERROR] Directory {path} does not exist.")
        return
    
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        if not os.path.isfile(imgPath):
            print(f"[WARN] File {imgPath} does not exist. Skipping...")
            continue
        
        image = cv2.imread(imgPath)
        if image is None:
            print(f"[WARN] Could not read image {imgPath}. Skipping...")
            continue
        
        name = os.path.splitext(img)[0]
        try:
            image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"[ERROR] Error resizing image {imgPath}: {e}")
            continue
        
        encs, locs = createEncodings(image)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)
        else:
            print(f"[WARN] No face found in image {imgPath}. Skipping...")
    
    saveEncodings(known_encodings, known_names, saveLocation)

# Process dataset images
def processDatasetImages(path="C:/Users/Admin/Desktop/Dataset", saveLocation="./dataset_encodings.pickle"):
    people_encodings, names = readEncodingsPickle("./known_encodings.pickle")
    
    if not os.path.isdir(path):
        print(f"[ERROR] Directory {path} does not exist.")
        return
    
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        if not os.path.isfile(imgPath):
            print(f"[WARN] File {imgPath} does not exist. Skipping...")
            continue
        
        image = cv2.imread(imgPath)
        if image is None:
            print(f"[WARN] Could not read image {imgPath}. Skipping...")
            continue
        
        orig = image.copy()
        try:
            image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"[ERROR] Error resizing image {imgPath}: {e}")
            continue
        
        encs, locs = createEncodings(image)
        
        if len(locs) > 1:
            saveImageToDirectory(orig, "Group", img)
            continue
        
        knownFlag = False
        for loc, unknown_encoding in zip(locs, encs):
            acceptBool, duplicateName, _ = compareFaceEncodings(unknown_encoding, people_encodings, names)
            if acceptBool:
                saveImageToDirectory(orig, duplicateName, img)
                knownFlag = True
                break
        
        if not knownFlag:
            saveImageToDirectory(orig, "Unknown", img)
        
        for loc in locs:
            top, right, bottom, left = loc
            cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
            cv2.imshow("Image", image)
            cv2.waitKey(1)
            cv2.destroyAllWindows()

def main():
    datasetPath = "C:/Users/Admin/Desktop/Dataset"
    peoplePath = "C:/Users/Admin/Desktop/people"
    processKnownPeopleImages(path=peoplePath)
    processDatasetImages(path=datasetPath)
    print("Completed")

if __name__ == "__main__":
    main()

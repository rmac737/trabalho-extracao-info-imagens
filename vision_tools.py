import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional

class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_single_face(self, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        img = cv2.imread(image_path)
        if img is None:
            return False, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) != 1:
            return False, None
        
        (x,y,w,h) = faces[0]
        face = img[y:y + h, x:x + w]
        return True, face
    
class ArcFaceEmbedder:
    def __init__(self, onnx_model_path: str, device: str = "cpu"):
        """
        onnx_model_path: caminho para o modelo ArcFace (ex: 'arcface.onnx')
        """
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 112, 112] normalmente

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        # BGR → RGB
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Redimensiona para 112x112
        img = cv2.resize(img, (112, 112)).astype(np.float32)

        # Normalização típica do ArcFace
        img = (img - 127.5) / 128.0

        # Agora NÃO transpõe — modelo espera NHWC
        # img shape: [112,112,3]

        # Expande para batch dimension
        img = np.expand_dims(img, axis=0)  # → [1,112,112,3]
        return img

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self.preprocess(face_bgr)
        emb = self.session.run(None, {self.input_name: inp})[0]
        emb = emb[0]  # (512,)
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        print(emb)
        return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    similaridade = float(np.dot(a, b))
    print(f"Similaridade {similaridade}")
    return similaridade


class FaceTools:
    def __init__(self, onnx_model_path: str, device: str = "cpu"):
        self.detector = FaceDetector()
        self.embedder = ArcFaceEmbedder(onnx_model_path, device=device)

    def detectar_rosto(self, image_path: str):
        ok, face = self.detector.detect_single_face(image_path)
        return {
            "ok": bool(ok),
            "message": "Um único rosto detectado." if ok else "Não há exatamente um rosto na imagem.",
        }

    def comparar_rostos(self, image1_path: str, image2_path: str, threshold: float = 0.75):
        ok1, face1 = self.detector.detect_single_face(image1_path)
        ok2, face2 = self.detector.detect_single_face(image2_path)

        if not ok1 or not ok2:
            return {
                "ok": False,
                "message": "Falha na detecção de rosto em uma ou ambas as imagens.",
                "similarity": None,
                "same_person": None,
            }

        emb1 = self.embedder.get_embedding(face1)
        emb2 = self.embedder.get_embedding(face2)
        sim = cosine_similarity(emb1, emb2)

        same = sim >= threshold
        return {
            "ok": True,
            "message": "Comparação realizada com sucesso.",
            "similarity": sim,
            "same_person": bool(same),
            "threshold": threshold,
        }

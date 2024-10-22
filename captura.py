import cv2
import time

def capture_video(output_path, duration=60):
    """
    Captura vídeo da câmera por um tempo definido e salva no caminho especificado.
    """
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Capturing Training Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Captura 60 segundos de vídeo para o treinamento
capture_video('training_video.avi', duration=60)

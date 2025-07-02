import cv2
import numpy as np
import improved_utils as utils
import sys
import os
from datetime import datetime

class DocumentScanner:
    def __init__(self):
        self.cap = None
        self.webCamFeed = False
        self.pathImage = ""
        self.heightImg = 640
        self.widthImg = 480
        self.count = 0
        self.running = True
        self.scan_folder = "Scanned"
        
        # Crear carpeta de escaneos si no existe
        if not os.path.exists(self.scan_folder):
            os.makedirs(self.scan_folder)
            print(f"Carpeta '{self.scan_folder}' creada.")
    
    def setup_input_method(self):
        """Configurar método de entrada (webcam o imagen)"""
        print("=" * 60)
        print("🔍 ESCÁNER DE DOCUMENTOS")
        print("=" * 60)
        print("\nOpciones disponibles:")
        print("1. Usar webcam/cámara IP")
        print("2. Usar imagen estática")
        print("3. Salir")
        
        while True:
            choice = input("\nSeleccione una opción (1-3): ").strip()
            
            if choice == '1':
                self.webCamFeed = True
                return self.setup_camera()
            elif choice == '2':
                self.webCamFeed = False
                return self.setup_image()
            elif choice == '3':
                print("Saliendo del programa...")
                return False
            else:
                print("❌ Opción inválida. Por favor, elija 1, 2 o 3.")
    
    def setup_camera(self):
        """Configurar cámara web o IP"""
        print("\n📷 Configuración de cámara:")
        print("1. Cámara local (0)")
        print("2. Cámara IP personalizada")
        print("3. Cámara IP por defecto (http://10.7.121.249:8080/video)")
        
        cam_choice = input("Seleccione opción de cámara (1-3): ").strip()
        
        if cam_choice == '1':
            camera_source = 0
        elif cam_choice == '2':
            camera_source = input("Ingrese la URL de la cámara IP: ").strip()
            if not camera_source:
                print("❌ URL vacía, usando cámara local.")
                camera_source = 0
        elif cam_choice == '3':
            camera_source = "http://10.7.121.249:8080/video"
        else:
            print("❌ Opción inválida, usando cámara local.")
            camera_source = 0
        
        return self.initialize_camera(camera_source)
    
    def initialize_camera(self, source):
        """Inicializar la cámara con manejo de errores"""
        print(f"🔄 Intentando conectar con: {source}")
        
        try:
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                print("❌ No se pudo acceder a la cámara especificada.")
                if source != 0:
                    print("🔄 Intentando con cámara local...")
                    self.cap = cv2.VideoCapture(0)
                    
                if not self.cap.isOpened():
                    print("❌ No se pudo acceder a ninguna cámara.")
                    print("🔄 Cambiando a modo imagen estática...")
                    self.webCamFeed = False
                    return self.setup_image()
            
            # Configurar propiedades de la cámara
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("✅ Cámara inicializada correctamente.")
            return True
            
        except Exception as e:
            print(f"❌ Error al inicializar cámara: {e}")
            return False
    
    def setup_image(self):
        """Configurar imagen estática"""
        print("\n🖼️  Configuración de imagen:")
        self.pathImage = input("Ingrese la ruta de la imagen (o presione Enter para '1.jpg'): ").strip()
        
        if not self.pathImage:
            self.pathImage = "1.jpg"
        
        if not os.path.exists(self.pathImage):
            print(f"❌ No se encontró la imagen: {self.pathImage}")
            return False
        
        print(f"✅ Imagen configurada: {self.pathImage}")
        return True
    
    def process_frame(self, img):
        """Procesar frame para detectar documento"""
        # Redimensionar imagen
        img = cv2.resize(img, (self.widthImg, self.heightImg))
        imgBlank = np.zeros((self.heightImg, self.widthImg, 3), np.uint8)
        
        # Preprocesamiento
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        
        # Obtener umbrales
        thres = utils.valTrackbars()
        imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
        
        # Operaciones morfológicas
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
        
        # Encontrar contornos
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        
        # Encontrar el contorno más grande
        biggest, maxArea = utils.biggestContour(contours)
        
        if biggest.size != 0:
            biggest = utils.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
            imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
            
            # Transformación de perspectiva
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [self.widthImg, 0], [0, self.heightImg], [self.widthImg, self.heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (self.widthImg, self.heightImg))
            
            # Recortar bordes
            imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored, (self.widthImg, self.heightImg))
            
            # Umbralización adaptativa
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
            
            imageArray = ([img, imgGray, imgThreshold, imgContours],
                         [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
            
            return imageArray, imgWarpColored
        else:
            imageArray = ([img, imgGray, imgThreshold, imgContours],
                         [imgBlank, imgBlank, imgBlank, imgBlank])
            return imageArray, None
    
    def save_scan(self, imgWarpColored):
        """Guardar escaneo con timestamp"""
        if imgWarpColored is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.scan_folder}/scan_{timestamp}_{self.count:03d}.jpg"
            
            cv2.imwrite(filename, imgWarpColored)
            print(f"✅ Escaneo guardado: {filename}")
            self.count += 1
            return filename
        return None
    
    def show_help_overlay(self, img):
        """Mostrar controles en pantalla"""
        overlay = img.copy()
        
        # Fondo semitransparente para el texto
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Texto de ayuda
        help_text = [
            "CONTROLES:",
            "S - Guardar escaneo",
            "H - Mostrar/ocultar ayuda",
            "Q/ESC - Salir",
            "SPACE - Pausa"
        ]
        
        for i, text in enumerate(help_text):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(img, text, (15, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
    
    def run(self):
        """Ejecutar el escáner"""
        if not self.setup_input_method():
            return
        
        utils.initializeTrackbars()
        show_help = True
        paused = False
        
        print("\n🚀 Escáner iniciado. Controles:")
        print("   S - Guardar escaneo")
        print("   H - Mostrar/ocultar ayuda")
        print("   SPACE - Pausar/reanudar")
        print("   Q/ESC - Salir")
        print("-" * 40)
        
        try:
            while self.running:
                if not paused:
                    if self.webCamFeed:
                        success, img = self.cap.read()
                        if not success:
                            print("❌ Error al capturar frame de la cámara")
                            break
                    else:
                        img = cv2.imread(self.pathImage)
                        if img is None:
                            print(f"❌ Error al cargar la imagen: {self.pathImage}")
                            break
                    
                    # Procesar imagen
                    imageArray, imgWarpColored = self.process_frame(img)
                    
                    # Labels para display
                    labels = [["Original", "Gris", "Umbral", "Contornos"],
                             ["Mayor Contorno", "Perspectiva", "Gris Corregido", "Umbral Adaptativo"]]
                    
                    # Crear imagen apilada
                    stackedImage = utils.stackImages(imageArray, 0.75, labels)
                    
                    # Mostrar ayuda si está habilitada
                    if show_help:
                        stackedImage = self.show_help_overlay(stackedImage)
                    
                    cv2.imshow("Document Scanner - Mejorado", stackedImage)
                
                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s') and not paused:
                    if imgWarpColored is not None:
                        filename = self.save_scan(imgWarpColored)
                        if filename:
                            # Mostrar confirmación visual
                            temp_img = stackedImage.copy()
                            cv2.rectangle(temp_img, 
                                        (int(temp_img.shape[1] / 2) - 200, int(temp_img.shape[0] / 2) - 50),
                                        (int(temp_img.shape[1] / 2) + 200, int(temp_img.shape[0] / 2) + 50),
                                        (0, 255, 0), cv2.FILLED)
                            cv2.putText(temp_img, "Scan Saved", 
                                      (int(temp_img.shape[1] / 2) - 100, int(temp_img.shape[0] / 2) + 10),
                                      cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
                            cv2.imshow("Document Scanner - Mejorado", temp_img)
                            cv2.waitKey(500)
                    else:
                        print("❌ No hay documento detectado para guardar")
                
                elif key == ord('h'):
                    show_help = not show_help
                    print(f"Ayuda {'activada' if show_help else 'desactivada'}")
                
                elif key == ord(' '):  # Espacio para pausar
                    paused = not paused
                    status = "pausado" if paused else "reanudado"
                    print(f"Escáner {status}")
                
                elif key == ord('q') or key == 27:  # 'q' o ESC
                    print("🛑 Cerrando escáner...")
                    self.running = False
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por el usuario")
        except Exception as e:
            print(f"❌ Error durante la ejecución: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpieza de recursos"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Recursos liberados. Total de escaneos: {self.count}")

# Función principal
def main():
    scanner = DocumentScanner()
    scanner.run()

if __name__ == "__main__":
    main()
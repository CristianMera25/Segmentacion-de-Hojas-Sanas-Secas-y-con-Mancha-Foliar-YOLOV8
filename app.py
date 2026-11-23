import threading
import time
from tkinter import filedialog
import customtkinter as ctk
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

CLASSES = ["Mancha", "Sana", "Seca"]
CLASS_COLORS = {
    "Sana": "#2ECC71",
    "Mancha": "#F39C12",
    "Seca": "#E74C3C",
}
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.45


class LeafSegDesktopApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("LeafSeg AI - Diagnóstico Fitosanitario")
        self.geometry("1120x680")
        self.minsize(980, 600)

        self.has_gpu = torch.cuda.is_available()
        self.device = 0 if self.has_gpu else "cpu"

        self.model = YOLO(MODEL_PATH)

        self.running = False
        self.cap = None
        self.raw_frame = None
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.fps = 0.0
        self.counts = {c: 0 for c in CLASSES}

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Panel izquierdo: visor de video
        self.video_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#121417")
        self.video_frame.grid(row=0, column=0, padx=(15, 8), pady=15, sticky="nsew")
        self.video_frame.grid_rowconfigure(1, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        top_bar = ctk.CTkFrame(self.video_frame, fg_color="transparent")
        top_bar.grid(row=0, column=0, padx=15, pady=(12, 5), sticky="ew")

        hw_text = (
            f"GPU: {torch.cuda.get_device_name(0)}"
            if self.has_gpu
            else "Modo: CPU"
        )
        ctk.CTkLabel(
            top_bar,
            text=hw_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#2ECC71" if self.has_gpu else "#F39C12",
        ).pack(side="left")

        self.lbl_fps = ctk.CTkLabel(
            top_bar,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#A0AAB5",
        )
        self.lbl_fps.pack(side="right")

        self.canvas_label = ctk.CTkLabel(
            self.video_frame,
            text="Presiona 'Iniciar Cámara' o carga una imagen/video",
            font=ctk.CTkFont(size=16),
            text_color="#6C757D",
        )
        self.canvas_label.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

        # Panel derecho: controles y conteo
        self.sidebar = ctk.CTkFrame(self, corner_radius=15)
        self.sidebar.grid(row=0, column=1, padx=(8, 15), pady=15, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar,
            text="LeafSeg AI",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(pady=(24, 2))

        ctk.CTkLabel(
            self.sidebar,
            text="Segmentación de Hojas en Vivero",
            font=ctk.CTkFont(size=13),
            text_color="#9AA0A6",
        ).pack(pady=(0, 20))

        src_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        src_frame.pack(fill="x", padx=18, pady=4)

        self.btn_cam = ctk.CTkButton(
            src_frame,
            text="Iniciar Cámara",
            command=self.start_camera,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_cam.pack(fill="x", pady=4)

        file_btns = ctk.CTkFrame(src_frame, fg_color="transparent")
        file_btns.pack(fill="x", pady=4)
        file_btns.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            file_btns,
            text="Imagen",
            command=self.load_image,
            height=36,
            fg_color="#2B303A",
            hover_color="#3B4252",
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(
            file_btns,
            text="Video",
            command=self.load_video,
            height=36,
            fg_color="#2B303A",
            hover_color="#3B4252",
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        self.btn_stop = ctk.CTkButton(
            src_frame,
            text="Detener",
            command=self.stop_stream,
            height=34,
            fg_color="#C0392B",
            hover_color="#E74C3C",
            state="disabled",
        )
        self.btn_stop.pack(fill="x", pady=(6, 4))

        ctk.CTkFrame(self.sidebar, height=2, fg_color="#2E3238").pack(
            fill="x", padx=18, pady=18
        )

        ctk.CTkLabel(
            self.sidebar,
            text="Hojas Detectadas",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(0, 10))

        self.count_labels = {}
        for cls_name in CLASSES:
            card = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#1E2227")
            card.pack(fill="x", padx=18, pady=5)

            ctk.CTkLabel(
                card,
                text=cls_name,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=CLASS_COLORS[cls_name],
            ).pack(side="left", padx=14, pady=10)

            lbl_val = ctk.CTkLabel(
                card,
                text="0",
                font=ctk.CTkFont(size=18, weight="bold"),
            )
            lbl_val.pack(side="right", padx=14, pady=10)
            self.count_labels[cls_name] = lbl_val

        health_card = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#181B20")
        health_card.pack(fill="x", padx=18, pady=(14, 10))

        self.lbl_health = ctk.CTkLabel(
            health_card,
            text="Índice de Salud: --%",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#A0AAB5",
        )
        self.lbl_health.pack(pady=10)

        ctk.CTkButton(
            self.sidebar,
            text="Guardar Diagnóstico",
            command=self.save_snapshot,
            height=38,
            fg_color="#1F6AA5",
            hover_color="#2980B9",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="bottom", fill="x", padx=18, pady=18)

    def apply_clahe(self, bgr_img):
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    def run_inference(self, frame):
        input_img = self.apply_clahe(frame)
        results = self.model.predict(
            source=input_img,
            conf=CONFIDENCE_THRESHOLD,
            device=self.device,
            verbose=False,
        )
        res = results[0]
        annotated = res.plot(boxes=True, masks=True)

        counts = {c: 0 for c in CLASSES}
        if res.boxes is not None:
            for cls_id in res.boxes.cls.cpu().numpy().astype(int):
                if 0 <= cls_id < len(CLASSES):
                    counts[CLASSES[cls_id]] += 1
        return annotated, counts

    @staticmethod
    def _is_green_or_empty(frame):
        if frame is None or frame.size == 0:
            return True
        b = float(frame[..., 0].mean())
        g = float(frame[..., 1].mean())
        r = float(frame[..., 2].mean())
        return (g > 100 and b < 30 and r < 30) or (b + g + r < 10)

    def start_camera(self):
        self.stop_stream()
        self.canvas_label.configure(image=None, text="Conectando cámara...")
        self.canvas_label.image = None
        self.update_idletasks()

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = False, None
            for _ in range(6):
                ret, frame = cap.read()
                if ret and not self._is_green_or_empty(frame):
                    break
                time.sleep(0.05)

            if ret and not self._is_green_or_empty(frame):
                self.cap = cap
                self._start_worker_threads(is_video_file=False)
                return
            cap.release()

        for url in ["http://127.0.0.1:4747/video", "http://localhost:4747/video"]:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and not self._is_green_or_empty(frame):
                    self.cap = cap
                    self._start_worker_threads(is_video_file=False)
                    return
                cap.release()

        self.canvas_label.configure(
            text="No se pudo obtener video de la cámara.\n"
                 "Verifica que esté conectada y transmitiendo."
        )

    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return
        self.stop_stream()
        self.cap = cv2.VideoCapture(path)
        self._start_worker_threads(is_video_file=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp")]
        )
        if not path:
            return
        self.stop_stream()
        img = cv2.imread(path)
        if img is not None:
            t0 = time.time()
            annotated, counts = self.run_inference(img)
            self.fps = 1.0 / max(time.time() - t0, 1e-5)
            self.counts = counts
            self.annotated_frame = annotated
            self._refresh_ui_elements()

    def _start_worker_threads(self, is_video_file=False):
        self.running = True
        self.raw_frame = None
        self.annotated_frame = None
        self.btn_stop.configure(state="normal")
        threading.Thread(
            target=self._capture_loop, args=(is_video_file,), daemon=True
        ).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        self._ui_loop()

    def _capture_loop(self, is_video_file):
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.01)
                continue

            with self.lock:
                self.raw_frame = frame

            if is_video_file:
                time.sleep(0.025)

    def _inference_loop(self):
        prev_time = time.time()
        while self.running:
            with self.lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            annotated, counts = self.run_inference(frame)
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-5)
            prev_time = now

            with self.lock:
                self.annotated_frame = annotated
                self.counts = counts
                self.fps = fps

    def _ui_loop(self):
        if not self.running:
            return
        self._refresh_ui_elements()
        self.after(25, self._ui_loop)

    def _refresh_ui_elements(self):
        with self.lock:
            frame = (
                self.annotated_frame.copy()
                if self.annotated_frame is not None
                else None
            )
            counts = self.counts.copy()
            fps = self.fps

        if frame is not None:
            w_box = max(self.canvas_label.winfo_width(), 640)
            h_box = max(self.canvas_label.winfo_height(), 480)
            h, w, _ = frame.shape
            scale = min(w_box / w, h_box / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            ctk_img = ctk.CTkImage(
                light_image=pil_img, dark_image=pil_img, size=(new_w, new_h)
            )
            self.canvas_label.configure(image=ctk_img, text="")
            self.canvas_label.image = ctk_img

        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
        total = sum(counts.values())
        for k, v in counts.items():
            self.count_labels[k].configure(text=str(v))

        if total > 0:
            health_pct = (counts["Sana"] / total) * 100
            color = (
                "#2ECC71"
                if health_pct >= 75
                else ("#F39C12" if health_pct >= 40 else "#E74C3C")
            )
            self.lbl_health.configure(
                text=f"Índice de Salud: {health_pct:.1f}%",
                text_color=color,
            )
        else:
            self.lbl_health.configure(text="Índice de Salud: --%", text_color="#A0AAB5")

    def save_snapshot(self):
        if self.annotated_frame is None:
            return
        filename = f"diagnostico_leafseg_{int(time.time())}.jpg"
        cv2.imwrite(filename, self.annotated_frame)
        self.lbl_health.configure(text=f"Guardado: {filename}", text_color="#3498DB")

    def stop_stream(self):
        self.running = False
        time.sleep(0.05)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop.configure(state="disabled")

    def on_close(self):
        self.stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = LeafSegDesktopApp()
    app.mainloop()
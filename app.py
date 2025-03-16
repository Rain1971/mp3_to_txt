import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import whisper
from pydub import AudioSegment
import tempfile
import torch
import queue

# Cola para comunicación entre hilos
log_queue = queue.Queue()

def detect_gpu():
    """Detecta si hay GPU disponible y devuelve información relevante"""
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'pytorch_version': torch.__version__,
        'details': ""
    }
    
    if gpu_info['available']:
        gpu_info['name'] = torch.cuda.get_device_name(0)
        gpu_info['count'] = torch.cuda.device_count()
        gpu_info['cuda_version'] = torch.version.cuda
        gpu_info['details'] = f"GPU detectada: {gpu_info['name']}\nNúmero de GPUs: {gpu_info['count']}\nVersión de CUDA: {gpu_info['cuda_version']}"
    else:
        gpu_info['details'] = "No se detectó ninguna GPU compatible. Se utilizará CPU para el procesamiento."
    
    return gpu_info

class WhisperGUI:
    def __init__(self, root, gpu_info):
        self.root = root
        self.root.title("Audio a Texto - Whisper")
        self.root.geometry("800x800")
        self.root.resizable(True, True)
        
        # Guardar información de GPU
        self.gpu_info = gpu_info
        
        # Variables
        self.input_folder = tk.StringVar(value="audios")
        self.output_folder = tk.StringVar(value="textos")
        self.model_size = tk.StringVar(value="tiny")
        self.language = tk.StringVar(value="ca")
        self.use_fp16 = tk.BooleanVar(value=False)
        self.force_cpu = tk.BooleanVar(value=False)
        self.is_processing = False
        self.processing_thread = None
        
        # Crear interfaz
        self.create_widgets()
        
        # Configurar cola de logs
        self.setup_queue_handler()
        
        # Mostrar información de GPU en el log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, "Información del sistema:\n")
        self.log_text.insert(tk.END, f"- Versión de PyTorch: {self.gpu_info['pytorch_version']}\n")
        self.log_text.insert(tk.END, f"- CUDA disponible: {self.gpu_info['available']}\n")
        if self.gpu_info['available']:
            self.log_text.insert(tk.END, f"- GPU detectada: {self.gpu_info['name']}\n")
            self.log_text.insert(tk.END, f"- Número de GPUs: {self.gpu_info['count']}\n")
            self.log_text.insert(tk.END, f"- Versión de CUDA: {self.gpu_info['cuda_version']}\n")
        else:
            self.log_text.insert(tk.END, "- No se detectó GPU. Se utilizará CPU.\n")
            self.log_text.insert(tk.END, "- Asegúrate de tener los drivers de NVIDIA y CUDA instalados correctamente.\n")
        self.log_text.insert(tk.END, "\nListo para procesar archivos de audio.\n")
        self.log_text.config(state=tk.DISABLED)
        
        # Actualizar estado
        device_text = "GPU" if self.gpu_info['available'] else "CPU"
        self.status_var.set(f"Listo - Usando {device_text}")
    
    def create_widgets(self):
        # Frame principal con padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Banner de GPU
        self.create_gpu_banner(main_frame)
        
        # Sección de carpetas
        folder_frame = ttk.LabelFrame(main_frame, text="Carpetas", padding="5")
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Carpeta de entrada
        ttk.Label(folder_frame, text="Carpeta de entrada:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(folder_frame, text="Examinar...", command=self.browse_input_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Carpeta de salida
        ttk.Label(folder_frame, text="Carpeta de salida:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(folder_frame, text="Examinar...", command=self.browse_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # Sección de opciones
        options_frame = ttk.LabelFrame(main_frame, text="Opciones", padding="5")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Modelo
        ttk.Label(options_frame, text="Modelo:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        model_combo = ttk.Combobox(options_frame, textvariable=self.model_size, values=["tiny", "base", "small", "medium", "large"], state="readonly")
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Idioma
        ttk.Label(options_frame, text="Idioma:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        language_entry = ttk.Entry(options_frame, textvariable=self.language, width=10)
        language_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(options_frame, text="(ca: catalán, es: español, en: inglés, ...)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Opciones avanzadas
        advanced_frame = ttk.Frame(options_frame)
        advanced_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # FP16
        fp16_check = ttk.Checkbutton(advanced_frame, text="Usar FP16 (más rápido en GPU)", variable=self.use_fp16)
        fp16_check.pack(side=tk.LEFT, padx=5)
        if not self.gpu_info['available']:
            fp16_check.config(state=tk.DISABLED)
        
        # Forzar CPU
        force_cpu_check = ttk.Checkbutton(advanced_frame, text="Forzar uso de CPU", variable=self.force_cpu)
        force_cpu_check.pack(side=tk.LEFT, padx=20)
        if not self.gpu_info['available']:
            force_cpu_check.config(state=tk.DISABLED)
        
        # Panel de log
        log_frame = ttk.LabelFrame(main_frame, text="Registro", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear widget de texto con scroll para los logs
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón de inicio
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Iniciar Procesamiento", command=self.start_processing)
        self.start_button.pack(side=tk.RIGHT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Listo")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("", 9, "italic"))
        status_label.pack(side=tk.LEFT, padx=5)
    
    def create_gpu_banner(self, parent):
        # Banner para mostrar si se detectó GPU
        banner_frame = ttk.Frame(parent)
        banner_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Estilo para el banner
        s = ttk.Style()
        s.configure('GPU.TLabel', background='#EAFFEA', foreground='green', font=('Arial', 11, 'bold'))
        s.configure('CPU.TLabel', background='#EFEFEF', foreground='#505050', font=('Arial', 11))
        
        # Crear el banner según tipo de dispositivo
        if self.gpu_info['available']:
            banner = ttk.Label(
                banner_frame, 
                text=f"✓ GPU detectada: {self.gpu_info['name']}",
                style='GPU.TLabel',
                anchor='center',
                padding=10
            )
        else:
            banner = ttk.Label(
                banner_frame, 
                text="No se ha detectado GPU - El procesamiento será más lento",
                style='CPU.TLabel',
                anchor='center',
                padding=10
            )
        
        banner.pack(fill=tk.X)
    
    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta de entrada")
        if folder:
            self.input_folder.set(folder)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if folder:
            self.output_folder.set(folder)
    
    def setup_queue_handler(self):
        def check_queue():
            try:
                while True:
                    message = log_queue.get_nowait()
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                    log_queue.task_done()
            except queue.Empty:
                pass
            finally:
                self.root.after(100, check_queue)
        
        self.root.after(100, check_queue)
    
    def log_message(self, message):
        log_queue.put(message)
    
    def start_processing(self):
        if self.is_processing:
            return
        
        # Obtener valores de los campos
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        model_size = self.model_size.get()
        language = self.language.get()
        use_fp16 = self.use_fp16.get()
        force_cpu = self.force_cpu.get()
        
        # Validar carpetas
        if not os.path.exists(input_folder):
            try:
                os.makedirs(input_folder)
                self.log_message(f"Se ha creado la carpeta de entrada: {input_folder}")
            except Exception as e:
                self.log_message(f"Error al crear la carpeta de entrada: {str(e)}")
                return
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                self.log_message(f"Se ha creado la carpeta de salida: {output_folder}")
            except Exception as e:
                self.log_message(f"Error al crear la carpeta de salida: {str(e)}")
                return
        
        # Iniciar procesamiento en un hilo separado
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.status_var.set("Procesando...")
        
        self.processing_thread = threading.Thread(
            target=self.process_audio_files,
            args=(input_folder, output_folder, model_size, language, use_fp16, force_cpu)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_audio_files(self, input_folder, output_folder, model_size, language, use_fp16, force_cpu):
        try:
            # Determinar dispositivo
            if force_cpu:
                device = "cpu"
                self.log_message("Forzando uso de CPU según configuración.")
            else:
                device = self.gpu_info['device']
            
            self.log_message(f"Utilizando dispositivo: {device}")
            
            # Mostrar información de precisión
            if use_fp16:
                if device == "cuda":
                    self.log_message("Usando precisión FP16 (media precisión)")
                else:
                    self.log_message("Advertencia: FP16 solo es útil con GPU. Se usará FP32 con CPU.")
                    use_fp16 = False
            else:
                self.log_message("Usando precisión FP32 (precisión completa)")
            
            # Buscar archivos de audio
            supported_formats = ['.mp3', '.wav', '.opus', '.ogg', '.flac', '.m4a']
            audio_files = []
            
            for file in os.listdir(input_folder):
                file_path = os.path.join(input_folder, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in supported_formats:
                        audio_files.append(file_path)
            
            if not audio_files:
                self.log_message(f"No se encontraron archivos de audio soportados en {input_folder}")
                self.finish_processing()
                return
            
            total_files = len(audio_files)
            self.log_message(f"Se encontraron {total_files} archivos de audio para procesar")
            
            # Cargar modelo Whisper (una sola vez)
            self.log_message(f"Cargando modelo Whisper ({model_size}) en dispositivo {device}...")
            model = whisper.load_model(model_size, device=device)
            
            # Procesar cada archivo
            for i, file_path in enumerate(audio_files):
                try:
                    # Actualizar progreso
                    progress_percent = (i / total_files) * 100
                    self.progress_var.set(progress_percent)
                    
                    filename = os.path.basename(file_path)
                    self.log_message(f"\nProcesando archivo {i+1}/{total_files}: {filename}")
                    
                    # Determinar si el archivo es grande
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Tamaño en MB
                    
                    if file_size > 10:
                        self.log_message("Archivo grande detectado, procesando por segmentos...")
                        transcript = self.process_large_audio(file_path, model, model_size, language, device, use_fp16)
                    else:
                        transcript = self.transcribe_with_model(file_path, model, language, use_fp16)
                    
                    # Guardar transcripción
                    base_filename = os.path.basename(os.path.splitext(file_path)[0])
                    output_file = os.path.join(output_folder, base_filename + '.txt')
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    
                    self.log_message(f"Transcripción guardada en: {output_file}")
                    
                except Exception as e:
                    self.log_message(f"Error al procesar {file_path}: {str(e)}")
                    continue
            
            # Finalizar
            self.progress_var.set(100)
            self.log_message("\n¡Procesamiento completado!")
            
        except Exception as e:
            self.log_message(f"Error en el procesamiento: {str(e)}")
        finally:
            self.finish_processing()
    
    def transcribe_with_model(self, file_path, model, language, fp16):
        """Transcribe usando un modelo ya cargado"""
        self.log_message(f"Transcribiendo audio en {language}" + (" usando FP16" if fp16 else "") + "...")
        result = model.transcribe(file_path, language=language, fp16=fp16)
        return result["text"]
    
    def process_large_audio(self, file_path, model, model_size, language, device, fp16, chunk_size=10*60*1000):
        """Procesa archivos de audio grandes dividiéndolos en segmentos."""
        # Cargar el audio
        audio = AudioSegment.from_file(file_path)
        duration = len(audio)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Si no es tan grande, procesarlo directamente
        if duration <= chunk_size:
            return self.transcribe_with_model(file_path, model, language, fp16)
        
        # Dividir en segmentos
        transcriptions = []
        segments = (duration // chunk_size) + (1 if duration % chunk_size > 0 else 0)
        
        for i in range(0, duration, chunk_size):
            self.log_message(f"Procesando segmento {i//chunk_size + 1}/{segments}...")
            
            # Extraer segmento
            segment = audio[i:min(i+chunk_size, duration)]
            
            # Guardar en archivo temporal
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_path = temp_file.name
                segment.export(temp_path, format=file_ext[1:])
            
            # Transcribir segmento
            segment_text = self.transcribe_with_model(temp_path, model, language, fp16)
            transcriptions.append(segment_text)
            
            # Eliminar archivo temporal
            os.unlink(temp_path)
        
        # Unir transcripciones
        return " ".join(transcriptions)
    
    def finish_processing(self):
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.status_var.set("Listo")

def main():
    # Detectar GPU antes de iniciar la interfaz
    gpu_info = detect_gpu()
    
    # Crear la interfaz con la información de GPU
    root = tk.Tk()
    app = WhisperGUI(root, gpu_info)
    root.mainloop()

if __name__ == "__main__":
    main()
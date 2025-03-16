import os
import argparse
import whisper
from pydub import AudioSegment
import tempfile
import torch

def transcribe_audio(file_path, model_size='base', language='ca', device=None, fp16=False):
    """
    Transcribe el archivo de audio utilizando el modelo Whisper.
    
    Args:
        file_path (str): Ruta al archivo de audio
        model_size (str): Tamaño del modelo Whisper (tiny, base, small, medium, large)
        language (str): Código del idioma (ca para catalán por defecto)
        device (str): Dispositivo a utilizar (cuda o cpu)
        fp16 (bool): Si se debe usar precisión FP16 (media precisión) en lugar de FP32
        
    Returns:
        str: Texto transcrito
    """
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    # Verificar el formato
    supported_formats = ['.mp3', '.wav', '.opus', '.ogg', '.flac', '.m4a']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in supported_formats:
        raise ValueError(f"Formato no soportado. Formatos admitidos: {', '.join(supported_formats)}")
    
    # Si no se especifica el dispositivo, detectar automáticamente
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Cargando modelo Whisper ({model_size}) en dispositivo {device}...")
    model = whisper.load_model(model_size, device=device)
    
    print(f"Transcribiendo audio en {language}" + (" usando FP16" if fp16 else "") + "...")
    
    # El parámetro fp16 va a la función transcribe, no a load_model
    result = model.transcribe(file_path, language=language, fp16=fp16)
    
    return result["text"]

def process_large_audio(file_path, model_size='base', language='ca', device=None, fp16=False, chunk_size=10*60*1000):
    """
    Procesa archivos de audio grandes dividiéndolos en segmentos.
    
    Args:
        file_path (str): Ruta al archivo de audio
        model_size (str): Tamaño del modelo Whisper
        language (str): Código del idioma (ca para catalán por defecto)
        device (str): Dispositivo a utilizar (cuda o cpu)
        fp16 (bool): Si se debe usar precisión FP16 (media precisión) en lugar de FP32
        chunk_size (int): Tamaño de cada segmento en milisegundos (default: 10 minutos)
        
    Returns:
        str: Texto transcrito completo
    """
    # Cargar el audio con pydub (detecta automáticamente el formato por la extensión)
    audio = AudioSegment.from_file(file_path)
    
    # Obtener la duración total en milisegundos
    duration = len(audio)
    
    # Obtener la extensión del archivo
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Si el audio es corto, procesarlo directamente
    if duration <= chunk_size:
        return transcribe_audio(file_path, model_size, language, device, fp16)
    
    # Dividir en segmentos si es muy largo
    transcriptions = []
    
    for i in range(0, duration, chunk_size):
        print(f"Procesando segmento {i//chunk_size + 1}/{(duration//chunk_size) + 1}...")
        
        # Extraer el segmento
        segment = audio[i:min(i+chunk_size, duration)]
        
        # Guardar el segmento en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_path = temp_file.name
            segment.export(temp_path, format=file_ext[1:])  # Elimina el punto de la extensión
        
        # Transcribir el segmento
        segment_text = transcribe_audio(temp_path, model_size, language, device, fp16)
        transcriptions.append(segment_text)
        
        # Eliminar el archivo temporal
        os.unlink(temp_path)
    
    # Unir todas las transcripciones
    return " ".join(transcriptions)

def get_available_device():
    """Determina si hay una GPU disponible y devuelve el dispositivo adecuado."""
    # Muestra más información de diagnóstico para identificar problemas con CUDA
    print("Información de PyTorch:")
    print(f"- Versión de PyTorch: {torch.__version__}")
    print(f"- CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"- GPU detectada: {gpu_name}")
        print(f"- Número de GPUs: {torch.cuda.device_count()}")
        print(f"- Versión de CUDA: {torch.version.cuda}")
        return device
    else:
        device = "cpu"
        print("- No se detectó GPU. Usando CPU.")
        print("- Asegúrate de tener los drivers de NVIDIA y CUDA instalados correctamente.")
        return device

def process_audio_folder(folder_path='audios', output_folder='textos', model_size='base', language='ca', device=None, fp16=False):
    """
    Procesa todos los archivos de audio en una carpeta.
    
    Args:
        folder_path (str): Ruta a la carpeta con archivos de audio
        output_folder (str): Ruta a la carpeta donde se guardarán las transcripciones
        model_size (str): Tamaño del modelo Whisper
        language (str): Código del idioma
        device (str): Dispositivo a utilizar (cuda o cpu)
        fp16 (bool): Si se debe usar precisión FP16 (media precisión) en lugar de FP32
    """
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe. Creándola...")
        os.makedirs(folder_path)
        print(f"Carpeta {folder_path} creada. Por favor, coloca tus archivos de audio allí.")
        return
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        print(f"La carpeta de salida {output_folder} no existe. Creándola...")
        os.makedirs(output_folder)
        print(f"Carpeta {output_folder} creada.")
    
    # Obtener lista de formatos soportados
    supported_formats = ['.mp3', '.wav', '.opus', '.ogg', '.flac', '.m4a']
    
    # Buscar archivos de audio en la carpeta
    audio_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_formats:
                audio_files.append(file_path)
    
    if not audio_files:
        print(f"No se encontraron archivos de audio soportados en {folder_path}")
        return
    
    # Procesar cada archivo
    print(f"Se encontraron {len(audio_files)} archivos de audio para procesar")
    
    for i, file_path in enumerate(audio_files):
        print(f"\nProcesando archivo {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        
        try:
            # Determinar si el archivo es grande
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Tamaño en MB
            
            if file_size > 10:
                print("Archivo grande detectado, procesando por segmentos...")
                transcript = process_large_audio(file_path, model_size, language, device, fp16)
            else:
                transcript = transcribe_audio(file_path, model_size, language, device, fp16)
            
            # Obtener solo el nombre base del archivo sin ruta ni extensión
            base_filename = os.path.basename(os.path.splitext(file_path)[0])
            
            # Guardar la transcripción en la carpeta de salida
            output_file = os.path.join(output_folder, base_filename + '.txt')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            print(f"Transcripción guardada en: {output_file}")
            
        except Exception as e:
            print(f"Error al procesar {file_path}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Convertir archivos de audio a texto usando Whisper')
    parser.add_argument('--folder', default='audios', 
                        help='Carpeta que contiene los archivos de audio (por defecto: audios)')
    parser.add_argument('--output', default='textos',
                        help='Carpeta donde se guardarán las transcripciones (por defecto: textos)')
    parser.add_argument('--model', default='tiny', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Modelo de Whisper a utilizar (por defecto: tiny)')
    parser.add_argument('--language', default='ca', 
                        help='Código del idioma para la transcripción (por defecto: ca para catalán)')
    parser.add_argument('--fp16', action='store_true',
                        help='Usar precisión FP16 (media precisión) en lugar de FP32. Recomendado para GPU')
    parser.add_argument('--cpu', action='store_true',
                        help='Forzar el uso de CPU incluso si hay GPU disponible')
    args = parser.parse_args()
    
    # Comprobar el dispositivo disponible
    device = "cpu" if args.cpu else get_available_device()
    print(f"Utilizando dispositivo: {device}")
    
    # Modo de precisión
    if args.fp16:
        if device == "cuda":
            print("Usando precisión FP16 (media precisión)")
        else:
            print("Advertencia: FP16 solo es útil con GPU. Se usará FP32 con CPU.")
    else:
        print("Usando precisión FP32 (precisión completa)")
    
    # Procesar todos los archivos en la carpeta
    process_audio_folder(args.folder, args.output, args.model, args.language, device, args.fp16)
    
    return 0

if __name__ == "__main__":
    exit(main())
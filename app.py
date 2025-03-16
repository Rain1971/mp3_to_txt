import whisper
import os

# Cargar el modelo google Whisper (puedes escoger entre: tiny, base, small, medium, large)
modelo = whisper.load_model("base")

# Carpeta de los mp3
carpeta_mp3 = "./audios"

# Carpeta on de los txt
carpeta_textos = "./textos"
os.makedirs(carpeta_textos, exist_ok=True)

# Recorrer todos los archivos de la carpeta audios
for fitxero in os.listdir(carpeta_mp3):
    if fitxero.lower().endswith(".mp3"):
        ruta_audio = os.path.join(carpeta_mp3, fitxero)
        print(f"Convirtiendo: {ruta_audio}")

        # Transcripcion del audio
        result = modelo.transcribe(ruta_audio, language="ca", fp16=False)

        # Guarda el resultado en un archivo de texto
        nombre_texto = fitxero.replace(".mp3", ".txt")
        ruta_texto = os.path.join(carpeta_textos, nombre_texto)

        with open(ruta_texto, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Arxiu guardado: {ruta_texto}")

print("Conversión completada!")

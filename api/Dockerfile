# Usar una imagen base oficial de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos al contenedor
COPY requirements.txt ./

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación al contenedor
COPY ./src /app/src

# Exponer el puerto en el que la aplicación correrá
EXPOSE 8080

# Comando para correr la aplicación
CMD ["fastapi", "run", "src/main.py", "--port", "8080"]

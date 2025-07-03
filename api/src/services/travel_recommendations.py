import pathlib
import pickle
from collections import defaultdict
import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler

SERVICES_PATH = pathlib.Path(__file__).parent
MODEL_PATH = SERVICES_PATH / "models" / "modelo_feedback.pkl"
CARPETA_RECOMENDACIONES = SERVICES_PATH / "recomendaciones"

# Carga los diccionarios de score y conteo desde el archivo
with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    score_destinos = defaultdict(float, data['score_destinos'])
    conteo_destinos = defaultdict(int, data['conteo_destinos'])

print("Modelo de feedback cargado exitosamente.")


# 1. CARGA Y LIMPIEZA DEL DATASET REAL

archivos_csv = glob.glob(os.path.join(CARPETA_RECOMENDACIONES, '*.csv'))
if not archivos_csv:
    raise FileNotFoundError(f'No se encontraron archivos CSV en la carpeta: {CARPETA_RECOMENDACIONES}')
dfs = [pd.read_csv(archivo) for archivo in archivos_csv]
df = pd.concat(dfs, ignore_index=True)

# Limpieza básica de nulos
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Desconocido')

# Normalización de variables numéricas (excluyendo IDs si existen)
cols_to_normalize = [col for col in num_cols if not col.lower().endswith('id')]
if cols_to_normalize:
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

print("DataFrame df cargado y limpio (solo datos reales). Listo para usar.")

from collections import defaultdict

# Diccionarios globales para score y conteo de ratings por destino
score_destinos = defaultdict(float)
conteo_destinos = defaultdict(int)
feedback_recomendaciones = []

def recomendar_por_usuario_y_preguntas(df, top_n=3):
    """
    Pregunta al usuario su ID y preferencias, recomienda destinos personalizados,
    pide calificación y guarda el feedback. Siempre muestra hasta top_n recomendaciones relajando filtros si es necesario.
    Usa el promedio de ratings dados por los usuarios para mejorar las recomendaciones populares.
    """
    import sys
    global score_destinos, conteo_destinos, feedback_recomendaciones  # Usa los globales cargados
    usuarios = df['UserID'].dropna().unique()
    print(f"Usuarios disponibles ({len(usuarios)}): {', '.join([str(int(u)) for u in usuarios[:10]])} ...")
    sys.stdout.flush()
    user_id = input("¿Cuál es tu ID de usuario? (elige uno de la lista): ").strip()
    try:
        user_id = float(user_id)
    except:
        print("ID inválido.")
        return

    print("\nResponde las siguientes preguntas para recibir una recomendación personalizada:\n")
    tipos_disp = df['Type'].dropna().unique()
    print(f"Tipos de destino disponibles: {', '.join(tipos_disp)}")
    print("Elige uno de los tipos anteriores. Ejemplo: Playa")
    sys.stdout.flush()
    tipo = input("¿Qué tipo de destino prefieres?: ").strip().capitalize()
    
    epocas_disp = df['BestTimeToVisit'].dropna().unique()
    print(f"Épocas del año disponibles: {', '.join(epocas_disp)}")
    print("Elige una de las épocas anteriores. Ejemplo: Nov-Feb")
    sys.stdout.flush()
    epoca = input("¿En qué época del año quieres viajar?: ").strip()
    
    estados_disp = df['State'].dropna().unique()
    print(f"Estados disponibles: {', '.join(estados_disp)}")
    print("Elige uno de los estados anteriores. Ejemplo: Goa")
    sys.stdout.flush()
    estado = input("¿Qué estado prefieres visitar?: ").strip().title()
    
    destinos_usuario = set(df[df['UserID'] == user_id]['DestinationID'].dropna().unique())
    # Filtro estricto (los 3 criterios)
    filtrado = df[
        (df['Type'].str.contains(tipo, case=False, na=False)) &
        (df['BestTimeToVisit'].str.contains(epoca, case=False, na=False)) &
        (df['State'].str.contains(estado, case=False, na=False))
    ]
    filtrado = filtrado[~filtrado['DestinationID'].isin(destinos_usuario)]
    nombres = list(filtrado['Name'].unique())
    # Si hay menos de top_n, relajar filtros
    if len(nombres) < top_n:
        # Solo tipo y estado
        filtrado2 = df[
            (df['Type'].str.contains(tipo, case=False, na=False)) &
            (df['State'].str.contains(estado, case=False, na=False))
        ]
        filtrado2 = filtrado2[~filtrado2['DestinationID'].isin(destinos_usuario)]
        for nombre in filtrado2['Name'].unique():
            if nombre not in nombres:
                nombres.append(nombre)
            if len(nombres) == top_n:
                break
    if len(nombres) < top_n:
        # Solo tipo
        filtrado3 = df[
            (df['Type'].str.contains(tipo, case=False, na=False))
        ]
        filtrado3 = filtrado3[~filtrado3['DestinationID'].isin(destinos_usuario)]
        for nombre in filtrado3['Name'].unique():
            if nombre not in nombres:
                nombres.append(nombre)
            if len(nombres) == top_n:
                break
    if len(nombres) < top_n:
        # Rellenar con los destinos mejor rankeados por rating de feedback, no visitados
        destinos_no_visitados = [d for d in df['Name'].unique() if d not in nombres and d not in df[df['UserID'] == user_id]['Name'].unique()]
        destinos_ordenados = sorted(
            destinos_no_visitados,
            key=lambda x: (score_destinos[x] / conteo_destinos[x]) if conteo_destinos[x] > 0 else 0,
            reverse=True
        )
        for nombre in destinos_ordenados:
            nombres.append(nombre)
            if len(nombres) == top_n:
                break
    nombres = nombres[:top_n]
    print("\nDestinos recomendados:", nombres)
    sys.stdout.flush()

    # Pedir calificación al usuario
    while True:
        calificacion = input("\n¿Te gustaron las recomendaciones? Califícalas del 1 (muy malas) al 5 (excelentes): ").strip()
        if calificacion in ['1', '2', '3', '4', '5']:
            calificacion = int(calificacion)
            break
        else:
            print("Por favor, ingresa un número del 1 al 5.")

    # Guardar feedback y actualizar score de destinos
    feedback_recomendaciones.append({
        'user_id': user_id,
        'recomendaciones': list(nombres),
        'calificacion': calificacion
    })
    for nombre in nombres:
        score_destinos[nombre] += calificacion
        conteo_destinos[nombre] += 1
    print("\n¡Gracias por tu calificación! Tu opinión ha sido registrada.")
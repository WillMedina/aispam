<?php
// --- Configuración de la Base de Datos ---
$db_host = 'localhost';
$db_name = '';
$db_user = '';
$db_pass = '';
$db_charset = 'utf8mb4';

$dsn = "mysql:host=$db_host;dbname=$db_name;charset=$db_charset";
$options = [
    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    PDO::ATTR_EMULATE_PREPARES => false,
];

$pdo = null; // Inicializar PDO fuera del bloque try para el finally
try {
    $pdo = new PDO($dsn, $db_user, $db_pass, $options);
} catch (\PDOException $e) {
    // En producción, loguear el error y mostrar un mensaje genérico
    die("Error de conexión a la base de datos: " . $e->getMessage());
}

// --- Lógica de Subida y Procesamiento del CSV ---
$uploadMessage = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['csv_file'])) {
    if ($_FILES['csv_file']['error'] === UPLOAD_ERR_OK) {
        $fileTmpPath = $_FILES['csv_file']['tmp_name'];
        $fileName = $_FILES['csv_file']['name'];

        $fileSize = $_FILES['csv_file']['size'];
        $fileType = $_FILES['csv_file']['type'];

        $fileNameCmps = explode(".", $fileName);
        $fileExtension = strtolower(end($fileNameCmps));

        $allowedfileExtensions = ['csv'];
        if (in_array($fileExtension, $allowedfileExtensions)) {

            $handle = null; // Inicializar handle para el finally
            try {
                $handle = fopen($fileTmpPath, "r");
                if ($handle === FALSE) {
                    throw new Exception("No se pudo abrir el archivo CSV para lectura.");
                }

                // 1. Leer cabeceras para determinar la estructura y el tipo de archivo para la metadata
                $cabeceras_crudas = fgetcsv($handle, 10000, ","); // Asume delimitador coma

                if ($cabeceras_crudas === FALSE) {
                    throw new Exception("El archivo CSV está vacío o no se pudo leer la fila de cabeceras.");
                }

                $cabeceras = array_map('trim', array_map('strtolower', $cabeceras_crudas));
                $col_mensaje_idx = array_search('mensaje', $cabeceras);
                $col_tipo_idx = array_search('tipo', $cabeceras);

                if ($col_mensaje_idx === false) {
                    throw new Exception("La columna 'mensaje' no fue encontrada en la cabecera del CSV.");
                }

                // Determinar el valor de 'tipo' para la tabla de metadata 'csv'
                // 1 si el CSV tiene columna 'tipo', 0 si no la tiene.
                $tipo_archivo_csv_para_metadata = ($col_tipo_idx !== false) ? 1 : 0;

                // --- Ahora que conocemos la estructura, podemos interactuar con la BD ---
                //$pdo->beginTransaction(); 
                // 2. Registrar metadata del CSV
                $user_ip = $_SERVER['REMOTE_ADDR'] ?? 'N/A';

                $stmt_meta = $pdo->prepare("CALL sp_insert_csv_metadata(?, ?)");
                // Pasamos el tipo determinado ($tipo_archivo_csv_para_metadata)
                $stmt_meta->execute([$user_ip, $tipo_archivo_csv_para_metadata]);
                $result_meta = $stmt_meta->fetch(PDO::FETCH_ASSOC);
                $csv_id = $result_meta['csv_id'];
                $stmt_meta->closeCursor();

                if (!$csv_id) {
                    throw new Exception("No se pudo obtener el ID del CSV principal.");
                }

                // 3. Procesar y registrar cada fila de datos del CSV
                $registros_procesados = 0;
                $fila_actual_datos = 0; // Para contar filas de datos, no la cabecera

                $stmt_registro = $pdo->prepare("CALL sp_insert_csv_registro(?, ?, ?)");

                while (($data = fgetcsv($handle, 10000, ",")) !== FALSE) {
                    $fila_actual_datos++;

                    $mensaje = $data[$col_mensaje_idx] ?? null;
                    if (empty(trim((string) $mensaje))) { // Convertir a string antes de trim para evitar error con null
                        // Opcional: saltar filas con mensaje vacío o registrar un error
                        // $uploadMessage .= "Advertencia: Fila de datos $fila_actual_datos sin mensaje. Omitida.<br>";
                        continue;
                    }

                    $tipo_recibido_normalizado = null; // Por defecto NULL
                    // Solo intentamos leer el tipo si la columna fue detectada en las cabeceras
                    if ($tipo_archivo_csv_para_metadata == 1 && isset($data[$col_tipo_idx])) {
                        $tipo_csv_valor_original = strtolower(trim($data[$col_tipo_idx]));
                        if ($tipo_csv_valor_original === 'spam' || $tipo_csv_valor_original === '1') {
                            $tipo_recibido_normalizado = 1; // Spam
                        } elseif ($tipo_csv_valor_original === 'ham' || $tipo_csv_valor_original === '0') {
                            $tipo_recibido_normalizado = 0; // Ham (no spam)
                        }
                        // Si es otro valor o está vacío después del trim, se queda como NULL
                    }

                    $stmt_registro->execute([$csv_id, $mensaje, $tipo_recibido_normalizado]);
                    $registros_procesados++;
                }

                // $pdo->commit(); 
                $uploadMessage = "Archivo CSV '<strong>{$fileName}</strong>' procesado con éxito.<br>";
                $uploadMessage .= "ID de carga CSV: <strong>{$csv_id}</strong>.<br>";
                $uploadMessage .= "Tipo de archivo (metadata): <strong>{$tipo_archivo_csv_para_metadata}</strong> (1=con columna tipo, 0=sin columna tipo).<br>";
                $uploadMessage .= "Total de registros procesados: <strong>{$registros_procesados}</strong>.<br>";
                $uploadMessage .= "<code>Exportacion JSON: <a target=\"_blank\" href=\"export_json.php?csv_id={$csv_id}\">Aqu&iacute;</a></code><br>";
                $uploadMessage .= "Resultados <a target=\"_blank\" href=\"resultados.php?csv_id={$csv_id}\">CSV_{$csv_id}</a>";
            } catch (Throwable $e) {
                /* if ($pdo && $pdo->inTransaction()) { // Solo rollback si la transacción se inició
                  //$pdo->rollBack();
                  } */
                $uploadMessage = "Error al procesar el archivo CSV: " . $e->getMessage();
            } finally {
                if ($handle !== null && is_resource($handle)) {
                    fclose($handle);
                }
            }
        } else {
            $uploadMessage = 'Error: Tipo de archivo no permitido. Solo se aceptan archivos .CSV.';
        }
    } else {
        $uploadMessage = 'Error al subir el archivo. Código de error: ' . $_FILES['csv_file']['error'] . ' - Intente nuevamente';
    }

    echo <<<JS
<script>
    /*Reescribe la URL sin recargar la página ni mantener el POST*/
    if (window.history.replaceState) {
        window.history.replaceState(null, null, window.location.pathname);
    }
</script>
JS;
}
?>

<!DOCTYPE html>
<html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Subir CSV</title>
        <style>
            body {
                font-family: sans-serif;
                margin: 20px;
                background-color: #f4f4f4;
                color: #333;
            }
            .container {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            input[type="submit"] {
                background-color: #007bff;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            .message {
                margin-top: 20px;
                padding: 10px;
                font-size: 18px;
                font-family: "Verdana", "Arial", sans-serif !important;
                border-radius: 4px;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Subir archivo CSV de Mensajes</h2>
            <p>El CSV debe tener la columna "mensaje". Opcionalmente puede incluir una columna "tipo".</p>
            <p>Si la columna "tipo" está presente, sus valores pueden ser: "spam", "ham", "1" (spam), "0" (ham).</p>

            <form action="apiCSV.php" method="post" enctype="multipart/form-data">
                <div>
                    <label for="csv_file">Selecciona el archivo CSV:</label>
                    <input type="file" name="csv_file" id="csv_file" accept=".csv" required>
                </div>
                <input type="submit" value="Subir y Procesar CSV">
            </form>

            <?php if (!empty($uploadMessage)): ?>
                <div class="message <?php echo (strpos($uploadMessage, 'Error') === 0 || strpos($uploadMessage, 'Advertencia') === 0) ? 'error' : 'success'; ?>">
                    <?php echo $uploadMessage; ?>
                </div>
            <?php endif; ?>
        </div>
    </body>
</html>
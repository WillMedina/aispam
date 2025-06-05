<?php

// --- Configuración de la Base de Datos (igual que antes) ---
$db_host = 'localhost';
$db_name = 'u691481820_aispamcsv';
$db_user = 'u691481820_aispamusr';
$db_pass = 'F&WpqbW!9';
$db_charset = 'utf8mb4';

$dsn = "mysql:host=$db_host;dbname=$db_name;charset=$db_charset";
$options = [
    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    PDO::ATTR_EMULATE_PREPARES => false,
];

// 1. Obtener y validar el csv_id de la URL (parámetro GET)
$csv_id_param = filter_input(INPUT_GET, 'csv_id', FILTER_VALIDATE_INT);

if ($csv_id_param === false || $csv_id_param <= 0) {
    http_response_code(400); // Bad Request
    header('Content-Type: application/json');
    echo json_encode([
        'error' => 'Parámetro csv_id inválido o faltante. Debe ser un entero positivo.'
            ], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
    exit;
}

try {
    $pdo = new PDO($dsn, $db_user, $db_pass, $options);
} catch (\PDOException $e) {
    http_response_code(500);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Error de conexión a la base de datos: ' . $e->getMessage()]);
    exit;
}

// --- Lógica para generar el JSON usando el Stored Procedure ---
try {
    // Preparar la llamada al Stored Procedure con el parámetro
    $stmt = $pdo->prepare("CALL sp_export_data_with_status(?)");
    $stmt->bindParam(1, $csv_id_param, PDO::PARAM_INT);
    $stmt->execute();

    // Obtener el primer conjunto de resultados (estado)
    $status_row = $stmt->fetch(PDO::FETCH_ASSOC);
    $sp_resultado = $status_row['resultado'] ?? 0;
    $sp_mensaje_guia = $status_row['mensaje_sp'] ?? 'Error al obtener estado del SP.';

    $output_data_final = []; // El JSON final será un array

    if ($sp_resultado == 1) {
        $stmt->nextRowset();
        $all_data_rows_for_csv = $stmt->fetchAll(PDO::FETCH_ASSOC);

        if (!empty($all_data_rows_for_csv)) {
            // Como ahora es para un solo csv_id, la estructura es más simple.
            // Tomamos la metadata de la primera fila (será la misma para todas las filas de este result set).
            $first_row = $all_data_rows_for_csv[0];
            $single_csv_output = [
                'metadata' => [
                    'csv_id' => $first_row['metadata_csv_id'],
                    'fecha_carga' => $first_row['metadata_fecha_carga'],
                    'ip_carga' => $first_row['metadata_ip_carga'],
                    'csv_original_incluia_columna_tipo' => (bool) $first_row['metadata_tipo_archivo_csv'],
                    'procesado' => (bool) $first_row['metadata_procesado'],
                    'activo' => (bool) $first_row['metadata_activo']
                ],
                'registros' => []
            ];

            foreach ($all_data_rows_for_csv as $row) {
                $registro_item = [
                    'registro_id' => $row['registro_id'],
                    'mensaje' => $row['cr_mensaje']
                ];

                if ($row['metadata_tipo_archivo_csv'] == 1) { // Si el CSV original tenía columna 'tipo'
                    $registro_item['tipo_recibido'] = $row['tipo_recibido'];
                    $registro_item['fecha_revision'] = $row['fecha_revision'];
                    $registro_item['probabilidad_spam'] = $row['probabilidad_spam'];
                    $registro_item['tfidf_generado'] = $row['tfidf_generado'];
                    $registro_item['modelo_usado'] = $row['modelo_usado'];
                }
                $single_csv_output['registros'][] = $registro_item;
            }
            // El resultado es un array que contiene el objeto del CSV solicitado
            $output_data_final[] = $single_csv_output;
        }
        // Si $all_data_rows_for_csv está vacío pero sp_resultado fue 1,
        // $output_data_final seguirá siendo [], lo cual es manejado por la lógica de sp_resultado.
    }
    // Si $sp_resultado es 0, $output_data_final permanece [], indicando que no hay datos.
    // El $sp_mensaje_guia puede ser usado para logging o debugging si se desea.

    $stmt->closeCursor();

    header('Content-Type: application/json');
    // Si $output_data_final está vacío (porque sp_resultado fue 0 o no se encontraron datos para el csv_id)
    // se enviará un array JSON vacío: []
    // Si se encontraron datos, será un array con un único objeto: [{...datos del csv...}]
    echo json_encode($output_data_final, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
} catch (PDOException $e) {
    //http_response_code(500);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Error al procesar la solicitud de datos para csv_id ' . $csv_id_param . ': ' . $e->getMessage()]);
}
?>
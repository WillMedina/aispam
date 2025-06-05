<?php

/**
 * botCSV.php
 *
 * Contiene la función procesarCsvId() para:
 *  1. Llamar a sp_export_data_no_processed(:id)
 *  2. Procesar el primer rowset (resultado, mensaje_sp)
 *  3. Si ok, obtener segundo rowset con datos (hasta 1000)
 *  4. Construir JSON con metadata y registros (fechas en 'Y-m-d H:i:s')
 *  5. Enviar JSON a API
 *  6. Procesar respuesta de API y actualizar BD via sp_update_csv_registro
 *
 * Si se ejecuta directamente por CLI, invoca procesarCsvId() con argv[1]
 */
// Configuración de BD
$dbHost = 'localhost';
$dbName = '';
$dbUser = '';
$dbPass = '';

/**
 * Procesa un CSV dado su ID: exporta, envía a API y actualiza BD
 *
 * @param int $dataId
 * @throws Exception on error
 */
function procesarCsvId(int $dataId): void {
    if ($dataId <= 0) {
        throw new InvalidArgumentException("ID inválido: $dataId");
    }

    // Conexión PDO
    $dsn = "mysql:host={$GLOBALS['dbHost']};dbname={$GLOBALS['dbName']};charset=utf8mb4";
    $pdo = new PDO($dsn, $GLOBALS['dbUser'], $GLOBALS['dbPass'], [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::MYSQL_ATTR_MULTI_STATEMENTS => true,
    ]);

    // 1. Ejecutar SP exportador
    $stmt = $pdo->prepare("CALL sp_export_data_no_processed(:id)");
    $stmt->bindParam(':id', $dataId, PDO::PARAM_INT);
    $stmt->execute();

    // 2. Primer rowset
    $first = $stmt->fetchAll();
    if (empty($first) || !isset($first[0]['resultado'])) {
        throw new RuntimeException("No se recibió resultado del SP.");
    }
    if ((int) $first[0]['resultado'] !== 1 || strtolower($first[0]['mensaje_sp'] ?? '') !== 'ok') {
        // Nada que procesar
        return;
    }

    // 3. Segundo rowset
    $stmt->nextRowset();
    $rows = $stmt->fetchAll() ?: [];
    if (empty($rows)) {
        // Conjunto existe pero sin registros
        return;
    }
    $rows = array_slice($rows, 0, 1000);

    // 4. Construir payload
    // Metadata
    $firstRow = $rows[0];
    $fechaCarga = null;
    try {
        $fechaCarga = (new DateTime($firstRow['metadata_fecha_carga']))->format('Y-m-d H:i:s');
    } catch (Exception $e) {
        
    }
    $metadata = [
        'csv_id' => (int) $firstRow['metadata_csv_id'],
        'fecha_carga' => $fechaCarga,
        'ip_carga' => $firstRow['metadata_ip_carga'],
        'csv_original_incluia_columna_tipo' => ($firstRow['metadata_tipo_archivo_csv'] == 1),
        'procesado' => ($firstRow['metadata_procesado'] == 1),
        'activo' => ($firstRow['metadata_activo'] == 1),
    ];

    // Registros
    $registros = [];
    foreach ($rows as $r) {
        $fechaRev = null;
        if (!empty($r['fecha_revision'])) {
            try {
                $fechaRev = (new DateTime($r['fecha_revision']))->format('Y-m-d H:i:s');
            } catch (Exception $e) {
                
            }
        }
        $registros[] = [
            'registro_id' => (int) $r['registro_id'],
            'mensaje' => $r['cr_mensaje'],
            'tipo_recibido' => isset($r['tipo_recibido']) ? (($r['tipo_recibido'] !== null) ? (int) $r['tipo_recibido'] : null) : null,
            'fecha_revision' => $fechaRev,
            'probabilidad_spam' => $r['probabilidad_spam'] !== null ? (float) $r['probabilidad_spam'] : null,
            'tfidf_generado' => $r['tfidf_generado'] !== null ? $r['tfidf_generado'] : null,
            'modelo_usado' => $r['modelo_usado'] !== null ? $r['modelo_usado'] : null,
        ];
    }

    $payload = [
        ['metadata' => $metadata, 'registros' => $registros]
    ];

    $stmt->closeCursor();

    $jsonPayload = json_encode($payload, JSON_UNESCAPED_UNICODE);

    // 5. Enviar a API
    $apiUrl = '';
    $ch = curl_init($apiUrl);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $jsonPayload,
    ]);
    $apiResponse = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($httpCode < 200 || $httpCode >= 300) {
        throw new RuntimeException("API returned HTTP $httpCode");
    }

    // 6. Procesar respuesta y actualizar BD
    $respData = json_decode($apiResponse, true);
    if (json_last_error() !== JSON_ERROR_NONE || !isset($respData[0]['registros'])) {
        throw new RuntimeException("Respuesta de API malformada");
    }

    $updateStmt = $pdo->prepare(
            "CALL sp_update_csv_registro(
            :csv_id,
            :registro_id,
            :tipo_recibido,
            :fecha_revision,
            :probabilidad_spam,
            :tfidf_generado,
            :modelo_usado
        )"
    );

    foreach ($respData as $item) {
        $csvId = $item['metadata']['csv_id'];
        foreach ($item['registros'] as $r) {
            $updateStmt->execute([
                ':csv_id' => $csvId,
                ':registro_id' => $r['registro_id'],
                ':tipo_recibido' => $r['tipo_recibido'],
                ':fecha_revision' => $r['fecha_revision'],
                ':probabilidad_spam' => $r['probabilidad_spam'],
                ':tfidf_generado' => $r['tfidf_generado'],
                ':modelo_usado' => $r['modelo_usado'],
            ]);
        }
    }
}

/*
// Si se ejecuta directamente vía CLI
if (PHP_SAPI === 'cli' && basename(__FILE__) === basename($_SERVER['SCRIPT_FILENAME'])) {
    $id = isset($argv[1]) ? (int) $argv[1] : 0;
    try {
        procesarCsvId($id);
        echo "botCSV: CSV ID=$id procesado.\n";
    } catch (Throwable $e) {
        fwrite(STDERR, "Error: {$e->getMessage()}\n");
        exit(1);
    }
} 
*/

// HTTP GET
if (PHP_SAPI !== 'cli') {
    $id = isset($_GET['id']) ? (int) $_GET['id'] : 0;
    header('Content-Type: application/json');
    try {
        procesarCsvId($id);
        echo json_encode(['status' => 'success', 'csv_id' => $id]);
    } catch (Throwable $e) {
        //http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
}

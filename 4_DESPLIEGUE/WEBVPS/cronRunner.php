<?php

/**
 * cronRunner.php
 *
 * Script para ejecutarse periódicamente por crontab. Hace:
 * 1. Llama al stored procedure sp_get_pending_csv_ids() para obtener csv_id pendientes.
 * 2. Por cada csv_id, incluye y ejecuta directamente la función procesarCsvId() de botCSV.php,
 *    de forma secuencial, con un retardo configurable entre cada ejecución para no saturar la API.
 *
 * Crontab ejemplo:
 * * /5 * * * * /usr/bin/php /ruta/a/cronRunner.php >> /var/log/cronRunner.log 2>&1
 */
// Configuración de BD
$dbHost = 'localhost';
$dbName = '';
$dbUser = '';
$dbPass = '';

// Retardo en segundos entre cada llamada para evitar saturar la API
declare(ticks=1);
$delaySeconds = 6;

// Incluir la versión refactorizada de botCSV.php
//require_once __DIR__ . '/botCSV.php';

function procesarCsvId(int $dataId, $pdo): void {
    if ($dataId <= 0) {
        throw new InvalidArgumentException("ID inválido: $dataId");
    }

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

    $updateStmt->closeCursor();
}

header('Content-Type: application/json');
try {
    // Conexión PDO
    $pdo = new PDO("mysql:host={$dbHost};dbname={$dbName};charset=utf8mb4", $dbUser, $dbPass, [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,]
    );

    // Llamar al SP para obtener CSV pendientes
    $stmt = $pdo->query("CALL sp_get_pending_csv_ids()");
    $pending = $stmt->fetchAll(PDO::FETCH_COLUMN);
    $stmt->closeCursor();

    if (empty($pending)) {
        //echo "No hay CSV pendientes.\n";
        echo json_encode(["resultado" => 1, "mensaje" => "No hay CSV pendientes"]);
        exit;
    }

    // Procesamiento secuencial con retardo
    foreach ($pending as $csvId) {
        //echo "Procesando CSV ID=$csvId...\n";
        try {
            procesarCsvId((int) $csvId, $pdo);
            //echo "CSV ID=$csvId procesado correctamente.\n";
            echo json_encode(["resultado" => 1, "mensaje" => "CSV ID=$csvId procesado correctamente."]);
        } catch (Throwable $e) {
            //fwrite(STDERR, "Error al procesar CSV ID=$csvId: {$e->getMessage()}\n");
            echo json_encode(["resultado" => 0, "mensaje" => "Error procesando CSV ID=$csvId.", "exception" => $e->getMessage()]);
        }
        // Retardo antes de la siguiente ejecución
        sleep($delaySeconds);
    }
} catch (PDOException $e) {
    //error_log('Error DB en cronRunner: ' . $e->getMessage());
    //exit("Error en la conexión a BD.\n");
    echo json_encode(["resultado" => 0, "mensaje" => "Error en la conexión a BD", "exception" => $e->getMessage()]);
} catch (Throwable $e) {
    //error_log('Error genérico en cronRunner: ' . $e->getMessage());
    //exit("Error inesperado.\n");
    echo json_encode(["resultado" => 0, "mensaje" => "Error genérico en cronRunner", "exception" => $e->getMessage()]);
}

<?php
/**
 * results.php
 *
 * Muestra estadísticas y resultados de un CSV procesado.
 */
// Configuración de BD
$dbHost = 'localhost';
$dbName = '';
$dbUser = '';
$dbPass = '';

// Umbral de probabilidad
define('THRESHOLD', 0.5);

// Obtener ID
$csvId = isset($_GET['csv_id']) ? (int) $_GET['csv_id'] : 0;
if ($csvId <= 0) {
    die("<div class='container'><div class='message error'>ID de CSV inválido</div></div>");
}

try {
    // Conexión PDO
    $pdo = new PDO("mysql:host=$dbHost;dbname=$dbName;charset=utf8mb4", $dbUser, $dbPass, [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::MYSQL_ATTR_MULTI_STATEMENTS => true,
    ]);

    // Llamar SP
    $stmt = $pdo->prepare("CALL sp_export_data_with_status(:id)");
    $stmt->bindParam(':id', $csvId, PDO::PARAM_INT);
    $stmt->execute();

    // Flags
    $flag = $stmt->fetch();
    $stmt->nextRowset();
    $rows = $stmt->fetchAll();

    if (!$flag) {
        die("<div class='container'><div class='message error'>CSV ID={$csvId} no existe.</div></div>");
    }
    if (empty($rows)) {
        die("<div class='container'><div class='message error'>CSV ID={$csvId} existe pero hay error con los datos.</div></div>");
    }

    // Metadata
    $meta = [
        'csv_id' => htmlspecialchars($rows[0]['metadata_csv_id']),
        'fecha_carga' => (new DateTime($rows[0]['metadata_fecha_carga']))->format('Y-m-d H:i:s'),
        'ip_carga' => htmlspecialchars($rows[0]['metadata_ip_carga']),
        'tipo_archivo_csv' => $rows[0]['metadata_tipo_archivo_csv'] == 1 ? 'Con Tipo' : 'Sin Tipo',
        'activo' => $rows[0]['metadata_activo'] == 1 ? 'Sí' : 'No',
    ];

    // Registros
    $records = [];
    $hasTrueLabels = true;
    $hasProb = true;
    foreach ($rows as $r) {
        $true = $r['tipo_recibido'] !== null ? (int) $r['tipo_recibido'] : null;
        $prob = $r['probabilidad_spam'] !== null ? (float) $r['probabilidad_spam'] : null;
        if ($true === null)
            $hasTrueLabels = false;
        if ($prob === null)
            $hasProb = false;
        $pred = ($prob !== null) ? ($prob >= THRESHOLD ? 1 : 0) : null;
        $records[] = [
            'registro_id' => (int) $r['registro_id'],
            'mensaje' => $r['cr_mensaje'],
            'true' => $true,
            'pred' => $pred,
            'fecha_revision' => $r['fecha_revision'],
            'cr_procesado' => $r['cr_procesado'] == 1 ? '&#x2705;' : '&#x274C;',
            'prob' => $prob,
            'tfidf' => htmlspecialchars($r['tfidf_generado']),
            'modelo' => htmlspecialchars($r['modelo_usado']),
        ];
    }

    // Métricas
    function computeMetrics($reals, $preds) {
        $tp = $tn = $fp = $fn = 0;
        for ($i = 0; $i < count($reals); $i++) {
            if ($reals[$i] == 1 && $preds[$i] == 1)
                $tp++;
            if ($reals[$i] == 0 && $preds[$i] == 0)
                $tn++;
            if ($reals[$i] == 0 && $preds[$i] == 1)
                $fp++;
            if ($reals[$i] == 1 && $preds[$i] == 0)
                $fn++;
        }
        $accuracy = ($tp + $tn) / max(1, ($tp + $tn + $fp + $fn));
        $precision = $tp / max(1, ($tp + $fp));
        $recall = $tp / max(1, ($tp + $fn));
        $f1 = 2 * $precision * $recall / max(1, ($precision + $recall));
        return compact('tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f1');
    }

} catch (Exception $e) {
    die("<div class='container'><div class='message error'>Error: {$e->getMessage()}</div></div>");
}
?>
<!DOCTYPE html>
<html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Resultados CSV <?= htmlspecialchars($csvId) ?></title>
        <style>
            body {
                font-family: "Verdana";
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
            h1, h2, h3 {
                margin-top: 0;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border:1px solid #ccc;
                padding:8px;
                text-align: left;
                font-size: 11px;
            }
            th {
                background-color: #f0f0f0;
            }
            .message {
                margin-top: 20px;
                padding: 10px;
                font-size: 18px;
                font-family: "Verdana", "Arial", sans-serif;
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

            .spam{
                font-weight: bold !important;
                color:red !important;
            }

            .ham{
                font-weight: bold !important;
                color:blue !important;
            }

            .na{
                color:gray !important;
            }

            .trues{
                background-color: #99ff99;
            }

            .falses{
                background-color: #ffcccc;
            }

            .mc{
                background-color: #ccccff;
                color: #000;
                border: 1px solid #6633ff;
            }

            .mc td, .mc th {
                font-size: 17px !important;
                font-weight: bold;
            }

            td code{
                font-size: 17px !important;
                font-weight: bold;
            }

            .table-responsive {
                overflow-x: auto;
                margin-bottom: 20px;
            }

            td.mensaje, td.tfidf {
                max-width: 200px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Resultados CSV <?= htmlspecialchars($csvId) ?></h1>
            <h2>Metadata</h2>
            <ul>
                <li><strong>CSV ID:</strong> <?= $meta['csv_id'] ?></li>
                <li><strong>Fecha Carga:</strong> <?= $meta['fecha_carga'] ?></li>
                <!-- <li><strong>IP Carga:</strong> <?= $meta['ip_carga'] ?></li> -->
                <li><strong>Tipo CSV:</strong> <?= $meta['tipo_archivo_csv'] ?></li>
                <!-- <li><strong>Activo:</strong> <?= $meta['activo'] ?></li> -->
            </ul>

            <?php
            if ($hasTrueLabels && $hasProb):
                $reals = array_column($records, 'true');
                $preds = array_column($records, 'pred');
                $metrics = computeMetrics($reals, $preds);
                ?>
                <div class="message mc">
                    <h2>Matriz de Confusión</h2>
                    <table>
                        <tr><th></th><th>Pred Ham (0)</th><th>Pred Spam (1)</th></tr>
                        <tr><th>Real Ham (0)</th><td class="trues"><?= $metrics['tn'] ?></td><td class="falses"><?= $metrics['fp'] ?></td></tr>
                        <tr><th>Real Spam (1)</th><td class="falses"><?= $metrics['fn'] ?></td><td class="trues"><?= $metrics['tp'] ?></td></tr>
                    </table>
                </div>
                <div class="message success">
                    <h3>Métricas</h3>
                    <ul>
                        <li><strong>Accuracy:</strong> <?= round($metrics['accuracy'], 4) ?></li>
                        <li><strong>Precision:</strong> <?= round($metrics['precision'], 4) ?></li>
                        <li><strong>Recall:</strong> <?= round($metrics['recall'], 4) ?></li>
                        <li><strong>F1-Score:</strong> <?= round($metrics['f1'], 4) ?></li>
                    </ul>
                </div>

            <?php elseif (!$hasTrueLabels): ?>
                <div class="message success">No se puede obtener matriz de confusi&oacute;n o m&eacute;tricas (datos reales faltantes en el env&iacute;o original).</div>
                <br>
            <?php else: ?>
                <div class="message error">Aún no se ha completado la evaluación (probabilidades faltantes).</div>
                <br>
            <?php endif; ?>

            <?php
            // Estadísticas adicionales
            $probs = array_filter(array_column($records, 'prob'), fn($v) => $v !== null);
            if (!empty($probs)):
                $count = count($probs);
                $mean = array_sum($probs) / $count;
                sort($probs);
                $median = $probs[floor($count / 2)];
                $above = count(array_filter($probs, fn($v) => $v >= THRESHOLD));
                $below = $count - $above;
                ?>
                <h2>Estadísticas de Probabilidades</h2>
                <ul>
                    <li><strong>Total con probabilidad:</strong> <?= $count ?></li>
                    <li><strong>Media:</strong> <?= round($mean, 4) ?></li>
                    <li><strong>Mediana:</strong> <?= round($median, 4) ?></li>
                    <li><strong>>= <?= THRESHOLD ?> (Spam):</strong> <?= $above ?></li>
                    <li><strong>< <?= THRESHOLD ?> (Ham):</strong> <?= $below ?></li>
                </ul>
            <?php endif; ?>

            <h2>Registros</h2>
            <table class="table-responsive">
                <tr>
                    <th>*</th>
                    <th>BDID</th>
                    <th>Mensaje</th>
                    <th>Tipo Recibido</th>
                    <th>Predicción</th>
                    <th>Fecha Revisión</th>
                    <th>Procesado</th>
                    <th>Prob. Spam</th>
                    <th>TF-IDF</th>
                    <th style="width: 10%">Modelo</th>
                </tr>
                <?php $conteo = 1; ?>
                <?php foreach ($records as $r): ?>
                    <tr>
                        <td><?= $conteo ?></td>
                        <td><?= $r['registro_id'] ?></td>
                        <td class="mensaje"><?= htmlspecialchars($r['mensaje']) ?></td>
                        <td><?= $r['true'] === null ? 'N/A' : ($r['true'] ? '<span class="spam">spam</span>' : '<span class="ham">ham</span>') ?></td>
                        <td><?= $r['pred'] === null ? 'N/A' : ($r['pred'] ? '<span class="spam">spam</span>' : '<span class="ham">ham</span>') ?></td>
                        <td><?= $r['fecha_revision'] ?></td>
                        <td><?= $r['cr_procesado'] ?></td>
                        <td><?= $r['prob'] ?></td>
                        <td class="tfidf"><?= $r['tfidf'] ?></td>
                        <td><?= $r['modelo'] ?></td>
                    </tr>
                    <?php $conteo++; ?>
                <?php endforeach; ?>
            </table>          
        </div>
    </body>
</html>

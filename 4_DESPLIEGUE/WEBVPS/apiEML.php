<?php

include 'dmail.php';

class EMLParser
{

    public static function extractTextFromEML($emlFile)
    {
        try {
            $message = file_get_contents($emlFile);
            $parts = explode("\r\n\r\n", $message, 2);
            $body = $parts[1];

            // Eliminar etiquetas HTML
            $textContent = strip_tags($body);

            return $textContent;
        } catch (Exception $e) {
            return "Error al procesar el archivo EML: " . $e->getMessage();
        }
    }
}

// Validar el archivo EML enviado
if (($_FILES['file']['error'] ?? '') === UPLOAD_ERR_OK) {
    $file = $_FILES['file'];
    $allowedExtensions = ['eml'];
    $allowedMimeTypes = ['message/rfc822'];

    // Validar la extensión del archivo
    $fileExtension = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
    if (!in_array($fileExtension, $allowedExtensions)) {
        header('Content-Type: application/json');
        echo json_encode(['resultado' => 0, 'message' => 'Extensión de archivo no válida. Debe ser .EML']);
        exit;
    }

    // Validar el tipo MIME del archivo
    $fileMimeType = mime_content_type($file['tmp_name']);
    if (!in_array($fileMimeType, $allowedMimeTypes)) {
        header('Content-Type: application/json');
        echo json_encode(['resultado' => 0, 'message' => 'Tipo de archivo no válido. Debe ser un archivo EML.']);
        exit;
    }

    // Procesar el archivo EML
    $emlFile = $file['tmp_name'];
    $textContent = EMLParser::extractTextFromEML($emlFile);

    header('Content-Type: application/json');
    //echo json_encode(['result' => 1, 'message' => 'Texto extraído correctamente', 'data' => $textContent]);
    $dmail = new dmail();
    $resultado = json_decode($dmail::checkText($textContent), true);

    echo json_encode(["resultado" => 1, 'mensaje' => 'ok', 'datos' => $resultado]);
    die();
} else {
    header('Content-Type: application/json');
    echo json_encode(['resultado' => 0, 'message' => 'Error al procesar el archivo EML']);
}

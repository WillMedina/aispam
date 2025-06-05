<?php

include 'dmail.php';

$datos = json_decode(file_get_contents('php://input'), true);
$texto = htmlentities($datos['texto'] ?? '');

header('Content-Type: application/json');

if (!isset($datos['texto']) or is_null($datos["texto"]) or strlen($texto) == 0) {
    echo json_encode(["resultado" => 0, 'mensaje' => 'Par&aacute;metros incorrectos', "datos" => $datos]);
    die();
}


$dmail = new dmail();
$resultado = json_decode($dmail::checkText($texto), true);

echo json_encode(["resultado" => 1, 'mensaje' => 'ok', 'datos' => $resultado]);
die();

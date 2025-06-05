<?php

// Importar la clase PHPMailer
use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;
use PHPMailer\PHPMailer\SMTP;

// Cargar las clases de PHPMailer
require 'phpmailer/src/Exception.php';
require 'phpmailer/src/PHPMailer.php';
require 'phpmailer/src/SMTP.php';

include 'dmail.php';

// Configuración de la conexión al servidor de correo electrónico
$hostname = '{/imap/ssl/novalidate-cert}INBOX';
$username = '';
$password = '';
//$logFile = '/home/bunkernorte/web/willmedina.me/dmail/logs/' . date('Ymd') . '.log';
$logFile = '/home/u691481820/domains/willmedina.me/public_html/aispam/logs' . date('Ymd') . '.log';

// Conectarse al servidor de correo electrónico
$inbox = imap_open($hostname, $username, $password) or die('No se pudo conectar al servidor de correo electrónico');

// Obtener los mensajes de correo electrónico no leídos
$emails = imap_search($inbox, 'UNSEEN');

if ($emails) {
    foreach ($emails as $email_number) {
        $logMessage1 = '[' . date('H:i:s') . '][' . $to . '][' . $subject . '][Se encontró correo no leído]' . PHP_EOL;
        file_put_contents($logFile, $logMessage1, FILE_APPEND);

        $structure = imap_fetchstructure($inbox, $email_number);
        $parts = $structure->parts;

        // Buscar la parte del mensaje que contiene el texto plano
        $textContent = '';
        foreach ($parts as $part) {
            if ($part->type == 0 && $part->subtype == 'PLAIN') {
                $textContent = imap_fetchbody($inbox, $email_number, $part->number);
                break;
            }
        }

        // Procesar el contenido del correo electrónico
        $result = procesarTexto($textContent);

        // Enviar una respuesta automática
        $overview = imap_fetch_overview($inbox, $email_number, 0);

        $to_full = $overview[0]->from;
        $emailStart = strstr($to_full, '<');
        $to = '';

        //esto para validar direcciones tipo "WILL MEDINA <wm@wm.m>"
        if ($emailStart !== false) {
            $to = substr($emailStart, 1, -1);
        } else {
            $to = trim($to_full);
        }

        $subject = 'Resultado del análisis de spam';
        $headers = "From: \r\n";
        $headers .= "Reply-To: aispam@willmedina.me\r\n";
        $headers .= "X-Mailer: PHP/" . phpversion();
        $headers .= "MIME-Version: 1.0\r\n";
        $headers .= "Content-Type: text/html; charset=UTF-8\r\n";
        //$body = "Hola,\n\nEl análisis del correo electrónico indica lo siguiente:\n\n " . $result . "\n\nSaludos,\nServicio de Detección de Spam";
        $body = "
        <html>
        <body>
            <p>Hola,</p>
            <p>El análisis del correo electrónico indica lo siguiente:</p>";
        $body .= $result;
        $body .= "<p>Saludos,<br>Servicio de Detección de Spam</p>
        </body>
        </html>";

        //mail($to, $subject, $body, $headers);
        enviarCorreo($to, $subject, $body);

        // Marcar el correo electrónico como leído
        imap_setflag_full($inbox, $email_number, '\\Seen');

        $logMessage = '[' . date('H:i:s') . '][' . $to . '][' . $subject . '][Correo procesado]' . PHP_EOL;
        file_put_contents($logFile, $logMessage, FILE_APPEND);
    }
}

imap_close($inbox);

function enviarCorreo($to, $subject, $body, $from = '', $fromName = 'Servicio automático de detección de SPAM') {

    // Crear una instancia de PHPMailer
    $mail = new PHPMailer(true);

    try {
        // Configuración del servidor SMTP
        $mail->SMTPDebug = 0;
        $mail->isSMTP();
        $mail->Host = 'smtp.hostinger.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'aispam@willmedina.me';
        $mail->Password = 'WwVbUpvj7[u';
        $mail->SMTPSecure = 'ssl';
        $mail->Port = 465;

        // Configurar la codificación de caracteres a UTF-8
        $mail->CharSet = 'UTF-8';

        // Detalles del correo electrónico
        $mail->setFrom($from, $fromName);
        $mail->addAddress($to);
        $mail->Subject = $subject;
        $mail->Body = $body;
        $mail->isHTML(true);
        $mail->AltBody = strip_tags($body);

        // Enviar el correo electrónico
        $mail->send();
        return true;
    } catch (Exception $e) {
        $logMessage = '[' . date('H:i:s') . '][' . $to . '][' . $subject . '][ERROR -> ' . $mail->ErrorInfo . ']' . PHP_EOL;
        file_put_contents($logFile, $logMessage, FILE_APPEND);
        return false;
    }
}

function procesarTexto($texto) {
    $dmail = new dmail();
    $resultado = json_decode($dmail::checkText($texto), true);
    $texto_final = '';

    if (is_array($resultado)) {
        //$porcentaje = number_format($resultado['confidence'] * 100, 2) . "%";
        $porcentaje = number_format($resultado["prediction_probability_spam"] * 100, 2) . '%';
        $spam_detectado = (($resultado['is_spam'] ? '<span style="color:si">SI</span>' : '<span style="color:green">NO</span>'));
        $texto_final = '<b>Los resultados obtenidos son:</b><br>';
        $texto_final .= "<b>Spam detectado: $spam_detectado</b><br>";
        $texto_final .= '<b>An&aacute;lisis de SPAM: <span style="color:blue">' . $porcentaje . '</span></b><br>';
        $texto_final .= '<b>Umbral de medida: <span style="color:blue">0.5 (o 50%)</span></b><br>';
    } else {
        $texto_final = 'Error analizando los datos de su correo electr&oacute;nico devido a problemas de servidor.<br />'
                . ' Intente enviarlo nuevamente m&aacute;s tarde.';
    }

    /*
      if ($resultado['resultado'] == '1') {

      } else {

      } */

    return $texto_final;
}

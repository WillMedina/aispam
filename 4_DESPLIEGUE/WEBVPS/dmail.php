<?php

class dmail
{

    static $api = '';

    private static function makeRequest($method, $url, $payload = null)
    {
        $ch = curl_init();

        $headers = [
            'Content-Type: application/json',
        ];

        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

        if ($method == 'POST') {
            curl_setopt($ch, CURLOPT_POST, 1);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
        }

        $response = curl_exec($ch);
        curl_close($ch);

        //return json_decode($response, true);
        return $response;
    }

    public static function checkText($text)
    {
        try {
            $payload = json_encode([
                'message' => $text
            ]);

            $respuesta = self::makeRequest("POST", self::$api, $payload);
            return $respuesta;
        } catch (Throwable $exc) {
            return [];
        }
    }
}

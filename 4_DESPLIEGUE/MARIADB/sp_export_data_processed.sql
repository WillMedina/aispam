DELIMITER $$
CREATE PROCEDURE `sp_export_data_processed`(
    IN `p_csv_id` INT
)
BEGIN
	DECLARE data_exists BOOLEAN DEFAULT FALSE;
	DECLARE EXIT HANDLER FOR SQLEXCEPTION
	BEGIN
		GET DIAGNOSTICS CONDITION 1 @sqlstate = RETURNED_SQLSTATE, 
		 @errno = MYSQL_ERRNO, @text = MESSAGE_TEXT;
		-- SET @full_error = CONCAT("ERROR ", @errno, " (", @sqlstate, "): ", @text);
		SELECT @errno as 'resultado', @sqlstate, @text;
	END;
    
    START TRANSACTION;

    -- Verificar si existen datos para el p_csv_id especificado
    IF EXISTS (SELECT 1
               FROM `csv` c
               JOIN `csv_registro` cr ON c.csv_id = cr.csv_id
               WHERE c.csv_id = p_csv_id
               LIMIT 1) THEN
        SET data_exists = TRUE;
    END IF;

    -- Primer conjunto de resultados: Estado de la operación
    IF data_exists THEN
        SELECT 1 AS resultado, 'OK' AS mensaje_sp;
    ELSE
        -- Verificar si el csv_id siquiera existe en la tabla csv, para un mensaje más específico
        IF EXISTS (SELECT 1 FROM `csv` WHERE `csv_id` = p_csv_id LIMIT 1) THEN
            SELECT 0 AS resultado, CONCAT('No se encontraron registros para el CSV ID: ', p_csv_id, ', aunque el CSV existe.') AS mensaje_sp;
        ELSE
            SELECT 0 AS resultado, CONCAT('No se encontró un CSV con ID: ', p_csv_id) AS mensaje_sp;
        END IF;
    END IF;

    -- Segundo conjunto de resultados: Los datos (solo si existen para ese p_csv_id)
    IF data_exists THEN
			SELECT
				1 AS resultado,
                'OK' AS mensaje,
				c.csv_id AS metadata_csv_id,
				c.fecha AS metadata_fecha_carga,
				c.ip AS metadata_ip_carga,
                c.procesado as metadata_procesado,
				c.tipo AS metadata_tipo_archivo_csv,
				c.activo AS metadata_activo,
				cr.registro_id,
				cr.mensaje as 'cr_mensaje',
				cr.tipo_recibido,
				cr.fecha_revision,
                cr.crontab_procesado as 'cr_procesado',
				cr.probabilidad_spam,
				cr.tfidf_generado,
				cr.modelo_usado
			FROM
				`csv` c
			JOIN
				`csv_registro` cr ON c.csv_id = cr.csv_id
			WHERE
				c.csv_id = p_csv_id -- Filtrar por el ID de metadata proporcionado
                AND cr.crontab_procesado = 1
			ORDER BY
				cr.registro_id ASC; -- Ordenar por registro_id ya que solo hay un csv_id
		END IF;
    COMMIT;
END$$
DELIMITER ;

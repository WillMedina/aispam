DELIMITER $$
CREATE PROCEDURE `sp_update_csv_registro`(
    IN p_csv_id INT,
    IN p_registro_id INT,
    IN p_tipo_recibido TINYINT,
    IN p_fecha_revision DATETIME,
    IN p_probabilidad_spam DOUBLE,
    IN p_tfidf_generado TEXT,
    IN p_modelo_usado TEXT
)
BEGIN
	DECLARE EXIT HANDLER FOR SQLEXCEPTION
	BEGIN
		GET DIAGNOSTICS CONDITION 1 @sqlstate = RETURNED_SQLSTATE, 
		 @errno = MYSQL_ERRNO, @text = MESSAGE_TEXT;
		-- SET @full_error = CONCAT("ERROR ", @errno, " (", @sqlstate, "): ", @text);
		SELECT @errno as 'resultado', @sqlstate, @text;
	END;
    
    START TRANSACTION;
		UPDATE csv_registro
		SET
			tipo_recibido     = p_tipo_recibido,
			fecha_revision    = p_fecha_revision,
			probabilidad_spam = p_probabilidad_spam,
			tfidf_generado    = p_tfidf_generado,
			modelo_usado      = p_modelo_usado,
			crontab_procesado = 1
		WHERE registro_id = p_registro_id
		  AND csv_id      = p_csv_id;
	COMMIT;
END$$
DELIMITER ;

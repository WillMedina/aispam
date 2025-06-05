DELIMITER $$
CREATE PROCEDURE `sp_insert_csv_registro`(
    IN `p_csv_id` INT,
    IN `p_mensaje` TEXT,
    IN `p_tipo_recibido` TINYINT -- Este es el tipo normalizado (0 para ham, 1 para spam)
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
			INSERT INTO `csv_registro` (`csv_id`, `mensaje`, `tipo_recibido`)
			VALUES (p_csv_id, p_mensaje, p_tipo_recibido);
			-- Los campos fecha_revision, probabilidad_spam, etc., se quedan NULL por defecto
    COMMIT;
END$$
DELIMITER ;

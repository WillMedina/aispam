DELIMITER $$
CREATE PROCEDURE `sp_insert_csv_metadata`(
	IN `p_ip` TEXT,
    IN `p_tipo` TINYINT -- Este 'tipo' es el de la tabla 'csv', no el de 'spam/ham'
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
			INSERT INTO `csv` (`ip`, `tipo`, `fecha`, `activo`)
			VALUES (p_ip, p_tipo, CURRENT_TIMESTAMP(), 1);

			SELECT LAST_INSERT_ID() AS csv_id;
    COMMIT;
END$$
DELIMITER ;

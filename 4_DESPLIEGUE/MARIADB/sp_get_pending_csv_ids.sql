DELIMITER $$
CREATE PROCEDURE `sp_get_pending_csv_ids`()
BEGIN
	DECLARE EXIT HANDLER FOR SQLEXCEPTION
	BEGIN
		GET DIAGNOSTICS CONDITION 1 @sqlstate = RETURNED_SQLSTATE, 
		 @errno = MYSQL_ERRNO, @text = MESSAGE_TEXT;
		-- SET @full_error = CONCAT("ERROR ", @errno, " (", @sqlstate, "): ", @text);
		SELECT @errno as 'resultado', @sqlstate, @text;
	END;
    
    START TRANSACTION;
		  SELECT DISTINCT csv_id
		  FROM csv_registro
		  WHERE crontab_procesado = 0;
	COMMIT;
END$$
DELIMITER ;

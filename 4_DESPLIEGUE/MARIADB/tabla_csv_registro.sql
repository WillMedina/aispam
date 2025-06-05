CREATE TABLE `csv_registro` (
  `registro_id` int(11) NOT NULL AUTO_INCREMENT,
  `csv_id` int(11) NOT NULL,
  `mensaje` text NOT NULL,
  `crontab_procesado` tinyint(4) NOT NULL DEFAULT 0,
  `tipo_recibido` tinyint(4) DEFAULT NULL,
  `fecha_revision` datetime DEFAULT NULL,
  `probabilidad_spam` double DEFAULT NULL,
  `tfidf_generado` text DEFAULT NULL,
  `modelo_usado` text DEFAULT NULL,
  PRIMARY KEY (`registro_id`)
) ENGINE=InnoDB AUTO_INCREMENT=6612 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

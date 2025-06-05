CREATE TABLE `csv` (
  `csv_id` int(11) NOT NULL AUTO_INCREMENT,
  `fecha` datetime NOT NULL DEFAULT current_timestamp(),
  `ip` text DEFAULT NULL,
  `tipo` tinyint(4) NOT NULL DEFAULT 0,
  `procesado` tinyint(4) NOT NULL DEFAULT 0,
  `activo` tinyint(4) NOT NULL DEFAULT 1,
  PRIMARY KEY (`csv_id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

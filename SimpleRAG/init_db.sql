-- Run this once in MySQL
CREATE DATABASE IF NOT EXISTS simple_rag
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Tables are created automatically by SQLAlchemy.
-- You can also create a dedicated user if you want:
-- CREATE USER 'rag_user'@'localhost' IDENTIFIED BY 'rag_pass';
-- GRANT ALL PRIVILEGES ON simple_rag.* TO 'rag_user'@'localhost';
-- FLUSH PRIVILEGES;
